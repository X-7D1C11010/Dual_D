"""Standalone training loop for the Dual_D algorithm.

Module purpose:
    Train the full Dual_D feature-level domain adaptation algorithm without
    importing original JMDA-Net scripts. The trainer owns dataset construction,
    model construction, epoch loops, validation, logging, and checkpointing.

Public interface:
    - run_training(args)

Expected args attributes:
    The entrypoint ``scripts/train_dual_d.py`` constructs these attributes from
    command-line flags and an optional JSON configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from dual_d.config import load_config
from dual_d.data import (
    M4SARClassificationDataset,
    MultiModalDomainDataset,
    PairedClassSampler,
    So2SatLCZ42Dataset,
    audit_dataset_splits,
    data_audit_errors,
    inverse_sqrt_class_weights,
    stratified_split_indices,
)
from dual_d.integration_adapter import DualDTrainingAdapter
from dual_d.models import (
    AISFeatureExtractor,
    Classifier,
    IRFeatureExtractor,
    LabelSmoothingCrossEntropy,
    OpticalResNet20Encoder,
    PlainMultimodalProjection,
    SARResNet20Encoder,
    TensorBasedAlignmentStable,
    VisualFeatureExtractor,
    set_requires_grad,
)
from dual_d.training.checkpointing import save_checkpoint, save_json
from dual_d.training.logging_utils import CSVMetricLogger, close_text_logger, setup_text_logger
from dual_d.training.metrics import classification_metrics


@dataclass
class ModelBundle:
    """Container for trainable model modules."""

    # Legacy names are retained in checkpoints.  For So2Sat, net_vis owns the
    # Optical/S2 encoder and net_ir owns the SAR/S1 encoder.
    net_vis: nn.Module
    net_ir: nn.Module
    net_ais: Optional[nn.Module]
    tal: nn.Module
    dual_adapter: DualDTrainingAdapter
    classifier: Classifier


class IndependentDomainLoaders:
    """Zip independently shuffled Source/Target loaders without pair matching.

    M4-SAR manifest rows retain physical correspondence for audit and later
    error analysis.  Training deliberately discards that correspondence: the
    two domains use distinct RandomSampler generators and are only joined by
    step number.
    """

    def __init__(
        self,
        source_dataset,
        target_dataset,
        args,
        seed: int | None = None,
    ) -> None:
        loader_kwargs = {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(getattr(args, "pin_memory", False)),
            "drop_last": True,
        }
        if int(args.num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(
                getattr(args, "persistent_workers", False)
            )
            loader_kwargs["prefetch_factor"] = int(
                getattr(args, "prefetch_factor", 2)
            )
        resolved_seed = int(args.seed if seed is None else seed)
        source_generator = torch.Generator().manual_seed(resolved_seed)
        target_generator = torch.Generator().manual_seed(resolved_seed + 1_000_003)
        self.source_loader = DataLoader(
            source_dataset,
            shuffle=True,
            generator=source_generator,
            **loader_kwargs,
        )
        self.target_loader = DataLoader(
            target_dataset,
            shuffle=True,
            generator=target_generator,
            **loader_kwargs,
        )

    def __iter__(self):
        return zip(self.source_loader, self.target_loader)

    def __len__(self) -> int:
        return min(len(self.source_loader), len(self.target_loader))


def parse_adaptive_batch_plan(value: str) -> tuple[tuple[float, int], ...]:
    """Parse ``free_gb:batch_size`` entries into an ordered batch plan."""

    entries = []
    for raw_entry in str(value).split(","):
        raw_entry = raw_entry.strip()
        if not raw_entry:
            continue
        try:
            raw_memory, raw_batch = raw_entry.split(":", maxsplit=1)
            memory_gb = float(raw_memory)
            batch_size = int(raw_batch)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Adaptive batch plan must use comma-separated free_gb:batch_size "
                f"entries, got {raw_entry!r}."
            ) from error
        if memory_gb <= 0 or batch_size <= 0:
            raise ValueError("Adaptive memory thresholds and batch sizes must be positive.")
        entries.append((memory_gb, batch_size))
    if not entries:
        raise ValueError("Adaptive batch plan must contain at least one entry.")
    entries.sort(key=lambda item: item[0])
    if len({memory for memory, _ in entries}) != len(entries):
        raise ValueError("Adaptive batch plan memory thresholds must be unique.")
    if any(
        entries[index][1] >= entries[index + 1][1]
        for index in range(len(entries) - 1)
    ):
        raise ValueError("Adaptive batch sizes must increase with available memory.")
    return tuple(entries)


def select_adaptive_batch_size(
    plan: tuple[tuple[float, int], ...],
    available_gb: float,
) -> int:
    """Choose the largest batch whose memory threshold is currently satisfied."""

    selected = plan[0][1]
    for threshold_gb, batch_size in plan:
        if float(available_gb) >= threshold_gb:
            selected = batch_size
        else:
            break
    return int(selected)


def reusable_cuda_memory_gb(device: torch.device) -> float:
    """Estimate memory usable by this process, including its CUDA reservation."""

    if device.type != "cuda":
        return float("inf")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    reusable_bytes = int(free_bytes) + int(torch.cuda.memory_reserved(device))
    return reusable_bytes / float(1024**3)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set common random seeds for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device_name: str) -> torch.device:
    """Resolve requested device name into a torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def validate_cuda_architecture(device: torch.device) -> None:
    """Fail early when the installed PyTorch build cannot run on the GPU.

    A cubin built for a lower minor compute capability is forward compatible
    with later GPUs in the same major family (for example, ``sm_86`` on
    ``sm_89``).  Requiring an exact token incorrectly rejects supported Ada
    GPUs, while a Blackwell ``sm_120`` device still requires an ``sm_12x``
    build.
    """

    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver and the active Python environment."
        )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    required_arch = f"sm_{major}{minor}"
    compiled_arches = set(torch.cuda.get_arch_list())
    compatible_arches = []
    for architecture in compiled_arches:
        token = architecture.removeprefix("sm_")
        if not token.isdigit() or len(token) < 2:
            continue
        compiled_major = int(token[:-1])
        compiled_minor = int(token[-1])
        if compiled_major == major and compiled_minor <= minor:
            compatible_arches.append(architecture)
    if compiled_arches and not compatible_arches:
        name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(
            "Incompatible PyTorch CUDA build: "
            f"{name} requires {required_arch}, but this installation provides "
            f"{', '.join(sorted(compiled_arches))}. Install a PyTorch/torchvision "
            f"build with an sm_{major}x target before starting training."
        )


def configure_visual_trainability(
    net_vis: VisualFeatureExtractor,
    freeze_visual_backbone: bool,
    pretrained_visual: bool,
) -> None:
    """Configure which visual extractor parameters should be trainable."""

    if freeze_visual_backbone and not pretrained_visual:
        warnings.warn(
            "freeze_visual_backbone=True with pretrained_visual=False would freeze "
            "random early ResNet layers. The full visual backbone will be trained.",
            RuntimeWarning,
        )
        freeze_visual_backbone = False

    if not freeze_visual_backbone:
        set_requires_grad(net_vis, True)
        return

    set_requires_grad(net_vis, False)
    trainable_prefixes = ("features.5", "features.6", "features.7", "proj")
    for name, parameter in net_vis.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad = True


def _set_frozen_batch_norm_eval(module: nn.Module) -> int:
    """Keep BatchNorm state fixed when all of that layer's parameters are frozen."""

    frozen_count = 0
    for child in module.modules():
        if not isinstance(child, _BatchNorm):
            continue
        parameters = list(child.parameters(recurse=False))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            child.eval()
            frozen_count += 1
    return frozen_count


def _probe_data_parallel(
    device: torch.device,
    device_ids: list[int],
    output_device: int,
) -> bool:
    """Check the NCCL broadcast used by ``DataParallel`` before wrapping models.

    Some multi-GPU installations can execute ordinary CUDA kernels but fail when
    NCCL performs the parameter broadcast used by DataParallel.  Detecting that
    condition here lets the training run continue on the primary GPU instead of
    failing on the first batch.
    """

    if len(device_ids) < 2:
        return True
    probe = nn.DataParallel(
        nn.Linear(8, 8, bias=False).to(device),
        device_ids=device_ids,
        output_device=output_device,
    )
    try:
        probe(torch.zeros(len(device_ids), 8, device=device))
        torch.cuda.synchronize(output_device)
    except RuntimeError as error:
        warnings.warn(
            "DataParallel/NCCL probe failed; continuing on the primary GPU. "
            "Try NCCL_P2P_DISABLE=1 (and, if needed, NCCL_SHM_DISABLE=1) "
            f"for multi-GPU execution. Original error: {error}",
            RuntimeWarning,
        )
        return False
    finally:
        del probe
        torch.cuda.empty_cache()
    return True


def build_datasets(args):
    """Build source train, target train, and target validation datasets."""

    if getattr(args, "dataset_type", "directory") == "so2sat_lcz42":
        return _build_so2sat_datasets(args)
    if getattr(args, "dataset_type", "directory") == "m4sar_classification":
        return _build_m4sar_datasets(args)

    source_train = MultiModalDomainDataset(
        root_dir=args.source_root,
        domain_type="source",
        phase=args.train_phase,
        layout=args.source_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        ais_folder=args.ais_folder,
        ais_root=getattr(args, "source_ais_root", "") or None,
        ais_data_path=getattr(args, "source_ais_data_path", "") or None,
        ais_match=args.ais_match,
        ais_sequence_length=args.ais_sequence_length,
        ais_normalize=args.ais_normalize,
        require_ais=getattr(args, "use_ais", False),
        image_size=args.image_size,
        resize_size=args.resize_size,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
        vis_augmentation_strength=getattr(args, "vis_augmentation_strength", None),
        ir_augmentation_strength=getattr(args, "ir_augmentation_strength", None),
        synchronize_modalities=getattr(args, "synchronize_modalities", False),
    )
    label_map = source_train.get_label_map()
    source_train_eval = MultiModalDomainDataset(
        root_dir=args.source_root,
        domain_type="source",
        phase=args.train_phase,
        layout=args.source_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        ais_folder=args.ais_folder,
        ais_root=getattr(args, "source_ais_root", "") or None,
        ais_data_path=getattr(args, "source_ais_data_path", "") or None,
        ais_match=args.ais_match,
        ais_sequence_length=args.ais_sequence_length,
        ais_normalize=args.ais_normalize,
        require_ais=getattr(args, "use_ais", False),
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        val_augment=False,
        train_augment=False,
        augmentation_strength=0.0,
        synchronize_modalities=getattr(args, "synchronize_modalities", False),
    )
    target_train = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.train_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        ais_folder=args.ais_folder,
        ais_root=getattr(args, "target_ais_root", "") or None,
        ais_data_path=getattr(args, "target_ais_data_path", "") or None,
        ais_match=args.ais_match,
        ais_sequence_length=args.ais_sequence_length,
        ais_normalize=args.ais_normalize,
        require_ais=getattr(args, "use_ais", False),
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
        vis_augmentation_strength=getattr(args, "vis_augmentation_strength", None),
        ir_augmentation_strength=getattr(args, "ir_augmentation_strength", None),
        synchronize_modalities=getattr(args, "synchronize_modalities", False),
    )
    target_val = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.val_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        ais_folder=args.ais_folder,
        ais_root=getattr(args, "target_ais_root", "") or None,
        ais_data_path=getattr(args, "target_ais_data_path", "") or None,
        ais_match=args.ais_match,
        ais_sequence_length=args.ais_sequence_length,
        ais_normalize=args.ais_normalize,
        require_ais=getattr(args, "use_ais", False),
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        val_augment=args.val_augment,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
        vis_augmentation_strength=getattr(args, "vis_augmentation_strength", None),
        ir_augmentation_strength=getattr(args, "ir_augmentation_strength", None),
        synchronize_modalities=getattr(args, "synchronize_modalities", False),
    )
    target_train_eval = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.train_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        ais_folder=args.ais_folder,
        ais_root=getattr(args, "target_ais_root", "") or None,
        ais_data_path=getattr(args, "target_ais_data_path", "") or None,
        ais_match=args.ais_match,
        ais_sequence_length=args.ais_sequence_length,
        ais_normalize=args.ais_normalize,
        require_ais=getattr(args, "use_ais", False),
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        val_augment=False,
        train_augment=False,
        augmentation_strength=0.0,
        synchronize_modalities=getattr(args, "synchronize_modalities", False),
    )
    return (
        source_train,
        source_train_eval,
        target_train,
        target_train_eval,
        target_val,
        None,
        label_map,
    )


def _build_m4sar_datasets(args):
    """Build official M4-SAR train/val splits without opening Target test."""

    common = {
        "manifest_path": args.m4sar_manifest,
        "data_root": getattr(args, "m4sar_data_root", "") or None,
        "input_size": int(getattr(args, "m4sar_input_size", 128)),
    }
    source_train = M4SARClassificationDataset(
        split="train",
        domain="source",
        augment=bool(getattr(args, "satellite_augment", True)),
        **common,
    )
    source_train_eval = source_train.with_domain("source", augment=False)
    target_train = source_train.with_domain(
        "target",
        augment=bool(getattr(args, "satellite_augment", True)),
    )
    target_train_eval = source_train.with_domain("target", augment=False)
    target_val = M4SARClassificationDataset(
        split="val",
        domain="target",
        augment=False,
        **common,
    )
    # Deliberately defer constructing the test dataset until checkpoint
    # selection is complete. This makes accidental test-driven selection
    # impossible inside the epoch loop.
    return (
        source_train,
        source_train_eval,
        target_train,
        target_train_eval,
        target_val,
        None,
        source_train.get_label_map(),
    )


def build_m4sar_test_dataset(args, domain: str = "target"):
    """Construct one M4-SAR test view only for post-selection evaluation."""

    return M4SARClassificationDataset(
        manifest_path=args.m4sar_manifest,
        data_root=getattr(args, "m4sar_data_root", "") or None,
        split="test",
        domain=domain,
        input_size=int(getattr(args, "m4sar_input_size", 128)),
        augment=False,
    )


def _build_so2sat_datasets(args):
    """Build leakage-safe So2Sat source/adaptation/test datasets."""

    dataset_root = Path(args.dataset_root)

    def path(name: str) -> Path:
        return dataset_root / str(getattr(args, name))

    common = {
        "sar_clip_after_normalize": getattr(
            args, "sar_clip_after_normalize", None
        )
    }
    source_data = path("so2sat_source_data")
    source_geo = path("so2sat_source_geo")
    target_adapt_data = path("so2sat_target_adapt_data")
    target_adapt_geo = path("so2sat_target_adapt_geo")
    target_test_data = path("so2sat_target_test_data")
    target_test_geo = path("so2sat_target_test_geo")

    target_index_dataset = So2SatLCZ42Dataset(
        target_adapt_data,
        target_adapt_geo,
        domain_role="target_adapt",
        augment=False,
        **common,
    )
    target_train_indices, target_val_indices = stratified_split_indices(
        target_index_dataset.labels,
        validation_fraction=float(args.target_adapt_val_fraction),
        seed=int(args.target_adapt_split_seed),
    )
    del target_index_dataset

    source_train = So2SatLCZ42Dataset(
        source_data,
        source_geo,
        domain_role="source",
        augment=bool(getattr(args, "satellite_augment", True)),
        **common,
    )
    source_train_eval = So2SatLCZ42Dataset(
        source_data,
        source_geo,
        domain_role="source",
        augment=False,
        **common,
    )
    target_train = So2SatLCZ42Dataset(
        target_adapt_data,
        target_adapt_geo,
        domain_role="target_adapt",
        indices=target_train_indices,
        augment=bool(getattr(args, "satellite_augment", True)),
        **common,
    )
    target_train_eval = So2SatLCZ42Dataset(
        target_adapt_data,
        target_adapt_geo,
        domain_role="target_adapt",
        indices=target_train_indices,
        augment=False,
        **common,
    )
    target_val = So2SatLCZ42Dataset(
        target_adapt_data,
        target_adapt_geo,
        domain_role="target_adapt",
        indices=target_val_indices,
        augment=False,
        **common,
    )
    target_test = So2SatLCZ42Dataset(
        target_test_data,
        target_test_geo,
        domain_role="target_test",
        augment=False,
        **common,
    )
    return (
        source_train,
        source_train_eval,
        target_train,
        target_train_eval,
        target_val,
        target_test,
        source_train.get_label_map(),
    )


def _audit_so2sat_protocol(
    source_train,
    target_train,
    target_val,
    target_test,
) -> Dict[str, object]:
    """Audit row-level separation and closed-set labels for So2Sat."""

    target_train_rows = set(int(value) for value in target_train.indices)
    target_val_rows = set(int(value) for value in target_val.indices)
    adapt_overlap = sorted(target_train_rows & target_val_rows)
    source_classes = sorted(set(int(value) for value in source_train.labels))
    target_train_classes = sorted(set(int(value) for value in target_train.labels))
    target_val_classes = sorted(set(int(value) for value in target_val.labels))
    target_test_classes = sorted(set(int(value) for value in target_test.labels))
    closed_set = (
        source_classes
        == target_train_classes
        == target_val_classes
        == target_test_classes
    )
    separate_test_file = (
        Path(target_test.data_path).resolve()
        != Path(target_train.data_path).resolve()
    )
    return {
        "dataset_type": "so2sat_lcz42",
        "source_data": str(Path(source_train.data_path).resolve()),
        "target_adapt_data": str(Path(target_train.data_path).resolve()),
        "target_test_data": str(Path(target_test.data_path).resolve()),
        "source_sample_count": len(source_train),
        "target_adapt_train_count": len(target_train),
        "target_adapt_val_count": len(target_val),
        "target_test_count": len(target_test),
        "target_adapt_row_overlap_count": len(adapt_overlap),
        "target_adapt_row_overlap_examples": adapt_overlap[:10],
        "target_test_is_separate_file": separate_test_file,
        "closed_set_17_classes": closed_set and len(source_classes) == 17,
        "source_classes": source_classes,
        "target_adapt_train_classes": target_train_classes,
        "target_adapt_val_classes": target_val_classes,
        "target_test_classes": target_test_classes,
        "leakage_detected": bool(adapt_overlap or not separate_test_file),
    }


def _audit_m4sar_protocol(source_train, target_train, target_val) -> Dict[str, object]:
    """Audit official M4-SAR roles without inspecting Target test."""

    source_pairs = [record.pair_id for record in source_train.records]
    target_pairs = [record.pair_id for record in target_train.records]
    source_labels = [int(record.label) for record in source_train.records]
    target_labels = [int(record.label) for record in target_train.records]
    train_views_match = source_pairs == target_pairs and source_labels == target_labels
    validation_pairs = {record.pair_id for record in target_val.records}
    split_overlap = sorted(set(source_pairs) & validation_pairs)
    train_source_scenes = {int(record.source_scene_id) for record in source_train.records}
    train_target_scenes = {int(record.target_scene_id) for record in source_train.records}
    validation_source_scenes = {
        int(record.source_scene_id) for record in target_val.records
    }
    validation_target_scenes = {
        int(record.target_scene_id) for record in target_val.records
    }
    source_scene_overlap = sorted(train_source_scenes & validation_source_scenes)
    target_scene_overlap = sorted(train_target_scenes & validation_target_scenes)
    valid_scene_partition = all(
        int(record.source_scene_id) <= 56087
        and int(record.target_scene_id) > 56087
        for record in (*source_train.records, *target_val.records)
    )
    return {
        "dataset_type": "m4sar_classification",
        "manifest": str(Path(source_train.manifest_path).resolve()),
        "source_train_count": len(source_train),
        "target_train_count": len(target_train),
        "target_val_count": len(target_val),
        "target_test_opened_during_selection": False,
        "physical_pair_metadata_retained": True,
        "pair_correspondence_used_for_training": False,
        "independent_source_target_shuffle": True,
        "train_domain_views_match_manifest": train_views_match,
        "source_scene_id_max_56087_and_target_above": valid_scene_partition,
        "train_validation_pair_overlap_count": len(split_overlap),
        "train_validation_pair_overlap_examples": split_overlap[:10],
        "train_validation_source_scene_overlap_count": len(source_scene_overlap),
        "train_validation_target_scene_overlap_count": len(target_scene_overlap),
        "leakage_detected": not (
            train_views_match
            and valid_scene_partition
            and not split_overlap
            and not source_scene_overlap
            and not target_scene_overlap
        ),
    }


def _model_mode(args) -> str:
    mode = str(getattr(args, "model_mode", "dual_d")).lower()
    allowed = {"dual_d", "optical_only", "sar_only", "simple_concat"}
    if mode not in allowed:
        raise ValueError(f"Unsupported model_mode={mode!r}; choose from {sorted(allowed)}.")
    return mode


def _training_component_status(args) -> Dict[str, object]:
    """Describe which optimization path is active for unambiguous artifacts."""

    mode = _model_mode(args)
    dual_d_active = mode == "dual_d"
    alignment_mode = str(getattr(args, "alignment_mode", "tal")).lower()
    translation_active = dual_d_active and bool(
        getattr(args, "translation_enabled", True)
    )
    return {
        "model_mode": mode,
        "training_objective": (
            "dual_d_domain_adaptation"
            if translation_active
            else (
                "aligned_source_target_classification_no_translation"
                if dual_d_active
                else "source_only_classification_baseline"
            )
        ),
        "train_acc_domain": "target" if dual_d_active else "source",
        "alignment_mode": alignment_mode,
        "tal_active": dual_d_active and alignment_mode == "tal",
        "translator_active": translation_active,
        "discriminators_active": translation_active,
        "module_c_active": translation_active
        and bool(getattr(args, "module_c_enabled", True)),
        "relation_drift_active": translation_active
        and bool(getattr(args, "modality_drift_enabled", True)),
    }


def _format_optional_metric(value, precision: int = 4) -> str:
    """Render disabled/not-evaluated metrics as n/a instead of a false zero."""

    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _apply_dual_loss_weight_overrides(dual_config, overrides):
    """Tune active loss weights while preserving ablation-config zeros."""

    dual_loss_weights = dict(overrides or {})
    for name, value in dual_loss_weights.items():
        current_value = float(getattr(dual_config.loss_weights, name))
        # The generated ablation config sets a removed constraint to zero.
        # Weather tuning may change active weights but must not re-enable it.
        if current_value != 0.0:
            setattr(dual_config.loss_weights, name, float(value))
    return dual_config


def _lr_scheduler_is_active(args, epoch: int) -> bool:
    """Return whether plateau schedulers may consume this epoch's metric."""

    start_epoch = max(int(getattr(args, "lr_scheduler_start_epoch", 1)), 1)
    return int(epoch) >= start_epoch


def _checkpoint_selection_is_eligible(args, epoch: int) -> bool:
    """Return whether this epoch may update best metrics and checkpoints."""

    start_epoch = max(
        int(getattr(args, "checkpoint_selection_min_epoch", 1)),
        1,
    )
    return int(epoch) >= start_epoch


def build_models(args, num_classes: int, device: torch.device) -> ModelBundle:
    """Instantiate all standalone Dual_D model modules."""

    dual_config = _apply_dual_loss_weight_overrides(
        load_config(args.dual_config),
        getattr(args, "dual_loss_weights", {}),
    )
    dataset_type = getattr(args, "dataset_type", "directory")
    model_mode = _model_mode(args)
    use_ais = bool(getattr(args, "use_ais", False))
    is_satellite = dataset_type in {"so2sat_lcz42", "m4sar_classification"}
    if is_satellite and use_ais:
        raise ValueError("AIS is not part of the satellite Optical-SAR pipeline.")
    num_modalities = 3 if use_ais else 2
    if model_mode == "dual_d":
        fused_dim = args.proj_dim * num_modalities
    elif model_mode == "simple_concat":
        fused_dim = args.feature_dim * num_modalities
    else:
        fused_dim = args.feature_dim
    if dual_config.feature_dim != fused_dim:
        dual_config.feature_dim = fused_dim
    dual_config.modality_dims = tuple(args.proj_dim for _ in range(num_modalities))

    if is_satellite:
        # Preserve legacy field/checkpoint names while assigning clear satellite
        # roles: net_vis=S2 Optical, net_ir=S1 SAR. Parameters are independent.
        net_vis = OpticalResNet20Encoder(
            input_channels=int(
                getattr(args, "optical_channels", 3)
                if dataset_type == "m4sar_classification"
                else getattr(args, "s2_channels", 10)
            ),
            output_dim=args.feature_dim,
        ).to(device)
        net_ir = SARResNet20Encoder(
            input_channels=int(
                getattr(args, "sar_channels", 1)
                if dataset_type == "m4sar_classification"
                else getattr(args, "s1_channels", 8)
            ),
            output_dim=args.feature_dim,
        ).to(device)
    else:
        net_vis = VisualFeatureExtractor(
            output_dim=args.feature_dim,
            pretrained=args.pretrained_visual,
        ).to(device)
        configure_visual_trainability(
            net_vis,
            args.freeze_visual_backbone,
            args.pretrained_visual,
        )
        net_ir = IRFeatureExtractor(output_dim=args.feature_dim).to(device)
    net_ais = None
    if use_ais:
        ais_sequence_length = int(
            getattr(args, "effective_ais_sequence_length", args.ais_sequence_length)
        )
        net_ais = AISFeatureExtractor(
            encoder_type=args.ais_encoder,
            sequence_length=ais_sequence_length,
            output_dim=args.feature_dim,
            dropout=args.ais_dropout,
        ).to(device)
    alignment_mode = str(getattr(args, "alignment_mode", "tal")).lower()
    if alignment_mode == "tal":
        tal = TensorBasedAlignmentStable(
            input_dims=[args.feature_dim] * num_modalities,
            output_dims=[args.proj_dim] * num_modalities,
            num_modalities=num_modalities,
        ).to(device)
    elif alignment_mode == "plain":
        tal = PlainMultimodalProjection(
            input_dims=[args.feature_dim] * num_modalities,
            output_dims=[args.proj_dim] * num_modalities,
        ).to(device)
    else:
        raise ValueError(f"Unsupported alignment_mode={alignment_mode!r}.")
    dual_adapter = DualDTrainingAdapter(dual_config).to(device)
    classifier = Classifier(
        input_dim=fused_dim,
        num_classes=num_classes,
        dropout=getattr(args, "classifier_dropout", 0.30),
    ).to(device)

    if model_mode != "dual_d":
        set_requires_grad(tal, False)
        set_requires_grad(dual_adapter, False)
        if model_mode == "optical_only":
            set_requires_grad(net_ir, False)
        elif model_mode == "sar_only":
            set_requires_grad(net_vis, False)
    elif not bool(getattr(args, "translation_enabled", True)):
        set_requires_grad(dual_adapter, False)

    # The feature extractors and classifier have ordinary tensor forward paths,
    # so they can use both server GPUs without wrapping the custom dual-loss
    # coordinator (whose auxiliary methods are not DataParallel forwards).
    if (
        bool(getattr(args, "multi_gpu", False))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        output_device = device.index if device.index is not None else 0
        device_ids = [output_device] + [
            index for index in range(torch.cuda.device_count()) if index != output_device
        ]
        if _probe_data_parallel(device, device_ids, output_device):
            net_vis = nn.DataParallel(net_vis, device_ids=device_ids, output_device=output_device)
            net_ir = nn.DataParallel(net_ir, device_ids=device_ids, output_device=output_device)
            if net_ais is not None:
                net_ais = nn.DataParallel(
                    net_ais,
                    device_ids=device_ids,
                    output_device=output_device,
                )
            classifier = nn.DataParallel(
                classifier,
                device_ids=device_ids,
                output_device=output_device,
            )
        else:
            warnings.warn(
                "multi_gpu was requested, but DataParallel was disabled after the "
                "NCCL probe. The run will use the primary CUDA device.",
                RuntimeWarning,
            )
    return ModelBundle(net_vis, net_ir, net_ais, tal, dual_adapter, classifier)


def _stable_monitor_score(
    values: List[float], monitor_mode: str, window: int
) -> float:
    """Return a higher-is-better score for recent validation stability."""

    if not values:
        raise ValueError("Validation monitor history must not be empty.")
    resolved_window = max(int(window), 1)
    if len(values) < resolved_window:
        return float("-inf")
    recent = values[-resolved_window:]
    if monitor_mode == "min":
        return -max(recent)
    return min(recent)


def build_optimizers(args, models: ModelBundle):
    """Build main and discriminator optimizers."""

    is_satellite = getattr(args, "dataset_type", "directory") in {
        "so2sat_lcz42",
        "m4sar_classification",
    }
    visual_params = [parameter for parameter in models.net_vis.parameters() if parameter.requires_grad]
    infrared_params = [
        parameter for parameter in models.net_ir.parameters() if parameter.requires_grad
    ]
    ais_params = (
        [parameter for parameter in models.net_ais.parameters() if parameter.requires_grad]
        if models.net_ais is not None
        else []
    )
    main_params = [
        *([] if is_satellite else infrared_params),
        *ais_params,
        *[parameter for parameter in models.tal.parameters() if parameter.requires_grad],
        *[
            parameter
            for parameter in models.dual_adapter.generator_parameters()
            if parameter.requires_grad
        ],
        *[parameter for parameter in models.classifier.parameters() if parameter.requires_grad],
    ]

    param_groups = []
    if is_satellite:
        encoder_params = [*visual_params, *infrared_params]
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": float(getattr(args, "lr_encoder", args.lr_main)),
                }
            )
    elif visual_params:
        param_groups.append({"params": visual_params, "lr": args.lr_visual})
    param_groups.append({"params": main_params, "lr": args.lr_main})

    optimizer_main = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    optimizer_disc = optim.AdamW(
        models.dual_adapter.discriminator_parameters(),
        lr=args.lr_discriminator,
        weight_decay=args.weight_decay,
    )
    return optimizer_main, optimizer_disc


def build_classification_criterion(
    args,
    source_dataset,
    device: torch.device,
    num_classes: int,
) -> nn.Module:
    """Build CE, using Source-train-only weights for M4-SAR when requested."""

    weights = None
    if (
        getattr(args, "dataset_type", "directory") == "m4sar_classification"
        and bool(getattr(args, "class_weighted_ce", True))
    ):
        configured = getattr(args, "class_weights", None)
        if configured:
            values = tuple(float(value) for value in configured)
        else:
            counts = [0 for _ in range(num_classes)]
            for label in source_dataset.labels:
                counts[int(label)] += 1
            values = inverse_sqrt_class_weights(counts)
        if len(values) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class weights, received {len(values)}."
            )
        weights = torch.tensor(values, dtype=torch.float32, device=device)
        args.effective_class_weights = [float(value) for value in values]
    else:
        args.effective_class_weights = None
    return LabelSmoothingCrossEntropy(
        eps=float(getattr(args, "label_smoothing", 0.0)),
        weight=weights,
    ).to(device)


def extract_fused_features(
    models: ModelBundle,
    source_batch: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    device: torch.device,
    args=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Dual_D or baseline features without changing core modules."""

    model_mode = _model_mode(args) if args is not None else "dual_d"
    if model_mode != "dual_d":
        feat_src = _encode_baseline_features(models, source_batch, device, model_mode)
        feat_tgt = _encode_baseline_features(models, target_batch, device, model_mode)
        return feat_src, feat_tgt, feat_src.new_zeros(())
    source_modalities = _encode_batch_modalities(models, source_batch, device)
    target_modalities = _encode_batch_modalities(models, target_batch, device)
    projected_source, projected_target, loss_tal = models.tal(
        source_modalities,
        target_modalities,
    )
    feat_src = torch.cat(projected_source, dim=1)
    feat_tgt = torch.cat(projected_target, dim=1)
    return feat_src, feat_tgt, loss_tal


def _encode_baseline_features(
    models: ModelBundle,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    model_mode: str,
) -> torch.Tensor:
    """Run only the front ends needed by the selected satellite baseline."""

    features = []
    if model_mode in {"sar_only", "simple_concat"}:
        features.append(models.net_ir(batch["sar"].to(device, non_blocking=True)))
    if model_mode in {"optical_only", "simple_concat"}:
        features.append(models.net_vis(batch["optical"].to(device, non_blocking=True)))
    if not features:
        raise ValueError(f"Unsupported baseline mode: {model_mode}")
    return features[0] if len(features) == 1 else torch.cat(features, dim=1)


def _encode_batch_modalities(
    models: ModelBundle,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Encode one legacy or So2Sat batch in a stable modality order."""

    if "sar" in batch and "optical" in batch:
        # Satellite order is S1 SAR first, S2 Optical second. This order defines
        # TAL projection blocks and therefore the Relation Drift split.
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        modalities = [models.net_ir(sar), models.net_vis(optical)]
    else:
        vis = batch["vis"].to(device, non_blocking=True)
        infrared = batch["ir"].to(device, non_blocking=True)
        modalities = [models.net_vis(vis), models.net_ir(infrared)]

    if models.net_ais is not None:
        if "ais" not in batch:
            raise RuntimeError(
                "AIS is enabled but the batch has no AIS tensor. Check AIS roots."
            )
        modalities.append(models.net_ais(batch["ais"].to(device, non_blocking=True)))
    return modalities


def _accumulate_logs(totals: Dict[str, float], logs: Dict[str, float]) -> None:
    """Accumulate slash-named log values into train metric totals."""

    for key, value in logs.items():
        metric_key = key.replace("/", "_")
        totals[metric_key] = totals.get(metric_key, 0.0) + float(value)


def _average_logged_metric(
    totals: Dict[str, float],
    key: str,
    divisor: float,
) -> float:
    """Return an averaged logged metric with a safe default."""

    return totals.get(key, 0.0) / max(divisor, 1.0)


def _adversarial_scale(args, epoch: int) -> float:
    """Return a linear adversarial warm-up scale in the closed interval [0, 1]."""

    warmup_epochs = max(int(getattr(args, "adversarial_warmup_epochs", 0)), 0)
    ramp_epochs = max(int(getattr(args, "adversarial_ramp_epochs", 0)), 0)
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min((epoch - warmup_epochs) / float(ramp_epochs), 1.0)


def _module_c_scale(args, epoch: int) -> float:
    """Return the warm-up/ramp scale for all Module-C constraints."""

    warmup_epochs = max(int(getattr(args, "module_c_warmup_epochs", 0)), 0)
    ramp_epochs = max(int(getattr(args, "module_c_ramp_epochs", 0)), 0)
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min((epoch - warmup_epochs) / float(ramp_epochs), 1.0)


def _modality_drift_scale(args, epoch: int) -> float:
    """Return the delayed ramp used by the modality-drift constraint."""

    warmup_epochs = max(
        int(getattr(args, "modality_drift_warmup_epochs", 0)),
        0,
    )
    ramp_epochs = max(
        int(getattr(args, "modality_drift_ramp_epochs", 0)),
        0,
    )
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min((epoch - warmup_epochs) / float(ramp_epochs), 1.0)


def _gradient_norm(parameters) -> float:
    """Compute the global L2 norm of currently populated gradients."""

    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            grad_norm = float(parameter.grad.detach().norm(2).cpu())
            squared_norm += grad_norm * grad_norm
    return squared_norm ** 0.5


def summarize_label_distribution(
    dataset,
    num_classes: int,
    label_map: Dict[str, int],
) -> Dict[str, object]:
    """Summarize class presence and imbalance for one dataset split."""

    id_to_label = {int(class_id): str(raw_label) for raw_label, class_id in label_map.items()}
    counts = [0 for _ in range(num_classes)]
    for label in getattr(dataset, "labels", []):
        class_id = int(label)
        if 0 <= class_id < num_classes:
            counts[class_id] += 1

    present_classes = [idx for idx, count in enumerate(counts) if count > 0]
    absent_classes = [idx for idx, count in enumerate(counts) if count == 0]
    return {
        "total_samples": int(sum(counts)),
        "num_classes": int(num_classes),
        "present_class_count": len(present_classes),
        "absent_class_count": len(absent_classes),
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "counts": [
            {
                "class_id": class_id,
                "raw_label": id_to_label.get(class_id, str(class_id)),
                "count": int(count),
            }
            for class_id, count in enumerate(counts)
        ],
    }


def _compact_distribution(summary: Dict[str, object]) -> str:
    """Format class counts for text logs."""

    counts = summary["counts"]
    parts = [
        f"{item['class_id']}:{item['count']}"
        for item in counts
        if int(item["count"]) > 0
    ]
    return ", ".join(parts) if parts else "none"


def _write_per_class_metrics(
    metric_logger: CSVMetricLogger,
    metrics: Dict[str, object],
    label_map: Dict[str, int],
    *,
    epoch: int,
    split: str,
    domain: str,
    checkpoint_selected: bool,
) -> None:
    """Write one auditable row per class for an evaluated split."""

    id_to_name = {int(class_id): name for name, class_id in label_map.items()}
    confusion = metrics["confusion_matrix"]
    for class_id in range(int(metrics["num_classes"])):
        metric_logger.write_row(
            {
                "epoch": int(epoch),
                "split": split,
                "domain": domain,
                "class_id": class_id,
                "class_name": id_to_name.get(class_id, str(class_id)),
                "accuracy_ovr": metrics["per_class_accuracy_ovr"][class_id],
                "precision": metrics["per_class_precision"][class_id],
                "recall": metrics["per_class_recall"][class_id],
                "f1": metrics["per_class_f1"][class_id],
                "specificity": metrics["per_class_specificity"][class_id],
                "correct": metrics["per_class_correct"][class_id],
                "support": metrics["per_class_support"][class_id],
                "predicted": int(sum(row[class_id] for row in confusion)),
                "became_best_checkpoint": bool(checkpoint_selected),
            }
        )


def _train_baseline_one_epoch(
    args,
    models: ModelBundle,
    training_loader,
    optimizer_main,
    criterion_cls: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, object]:
    """Train a source-only M4-SAR baseline; Target batches remain unused."""

    mode = _model_mode(args)
    models.net_vis.train(mode in {"optical_only", "simple_concat"})
    models.net_ir.train(mode in {"sar_only", "simple_concat"})
    models.tal.eval()
    models.dual_adapter.eval()
    models.classifier.train()
    loss_total = 0.0
    correct = 0
    sample_total = 0
    grad_norm_total = 0.0
    clipped_steps = 0
    steps = 0
    main_parameters = [
        parameter
        for group in optimizer_main.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]

    for source_batch, _target_batch in training_loader:
        labels = source_batch["label"].to(device, non_blocking=True)
        optimizer_main.zero_grad(set_to_none=True)
        features = _encode_baseline_features(models, source_batch, device, mode)
        logits = models.classifier(features)
        loss = criterion_cls(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite baseline loss at epoch={epoch}, step={steps + 1}."
            )
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(main_parameters, args.grad_clip)
                .detach()
                .cpu()
            )
        else:
            grad_norm = _gradient_norm(main_parameters)
        if args.grad_clip > 0 and grad_norm > args.grad_clip:
            clipped_steps += 1
        optimizer_main.step()

        correct += int((torch.argmax(logits.detach(), dim=1) == labels).sum().item())
        sample_total += int(labels.numel())
        loss_total += float(loss.detach().cpu())
        grad_norm_total += grad_norm
        steps += 1

    divisor = max(steps, 1)
    accuracy = correct / max(sample_total, 1)
    metrics = {
        **_training_component_status(args),
        "epoch": epoch,
        "train_loss": loss_total / divisor,
        "train_loss_cls": loss_total / divisor,
        "train_loss_cls_source": loss_total / divisor,
        "train_loss_cls_target": None,
        "train_loss_tal": None,
        "train_loss_dual_g": None,
        "train_loss_dual_d": None,
        "train_grad_norm_main": grad_norm_total / divisor,
        "train_grad_norm_discriminator": None,
        "train_grad_clip_fraction_main": clipped_steps / divisor,
        "train_grad_clip_fraction_discriminator": None,
        "train_discriminator_steps": 0.0,
        "train_adversarial_scale": None,
        "train_module_c_scale": None,
        "train_modality_drift_scale": None,
        "train_acc": accuracy,
        "train_acc_source": accuracy,
        "train_acc_target": None,
        "train_acc_source_like": None,
        "train_acc_target_like": None,
    }
    for name in (
        "discriminator_total",
        "discriminator_primary",
        "discriminator_auxiliary",
        "generator_total",
        "adv_primary",
        "adv_auxiliary",
        "cycle",
        "identity",
        "contrastive",
        "prototype_contrastive",
        "classification_feedback",
        "modality_drift",
        "modality_alignment_source_before",
        "modality_alignment_target_like_after",
        "modality_drift_source_to_target_raw",
        "modality_alignment_target_before",
        "modality_alignment_source_like_after",
        "modality_drift_target_to_source_raw",
        "weighted_cycle",
        "weighted_identity",
        "weighted_contrastive",
        "weighted_prototype_contrastive",
        "weighted_classification_feedback",
        "weighted_modality_drift",
    ):
        metrics[f"train_dual_d_{name}"] = None
    return metrics


def train_one_epoch(
    args,
    models: ModelBundle,
    paired_loader,
    optimizer_main,
    optimizer_disc,
    criterion_cls: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
) -> Dict[str, float]:
    """Train all Dual_D components for one epoch."""

    if _model_mode(args) != "dual_d":
        return _train_baseline_one_epoch(
            args,
            models,
            paired_loader,
            optimizer_main,
            criterion_cls,
            device,
            epoch,
        )

    models.net_vis.train()
    if bool(getattr(args, "freeze_frozen_batch_norm_stats", False)):
        _set_frozen_batch_norm_eval(models.net_vis)
    models.net_ir.train()
    if models.net_ais is not None:
        models.net_ais.train()
    models.tal.train()
    translation_enabled = bool(getattr(args, "translation_enabled", True))
    models.dual_adapter.train(translation_enabled)
    models.classifier.train()
    adversarial_scale = _adversarial_scale(args, epoch) if translation_enabled else 0.0
    module_c_scale = (
        _module_c_scale(args, epoch)
        if translation_enabled and bool(getattr(args, "module_c_enabled", True))
        else 0.0
    )
    modality_drift_scale = (
        _modality_drift_scale(args, epoch)
        if translation_enabled and bool(getattr(args, "modality_drift_enabled", True))
        else 0.0
    )

    totals = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_cls_source": 0.0,
        "loss_cls_target": 0.0,
        "loss_tal": 0.0,
        "loss_dual_g": 0.0,
        "loss_dual_d": 0.0,
        "source_correct": 0.0,
        "target_correct": 0.0,
        "source_like_correct": 0.0,
        "target_like_correct": 0.0,
        "sample_total": 0.0,
        "steps": 0.0,
        "disc_steps": 0.0,
        "grad_norm_main": 0.0,
        "grad_norm_disc": 0.0,
        "grad_clip_main_steps": 0.0,
        "grad_clip_disc_steps": 0.0,
    }

    for step, (source_batch, target_batch) in enumerate(paired_loader, start=1):
        source_labels = source_batch["label"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)
        labels_for_contrast = source_labels

        feat_src, feat_tgt, loss_tal = extract_fused_features(
            models,
            source_batch,
            target_batch,
            device,
            args,
        )
        dual_outputs = (
            models.dual_adapter.forward_features(
                feat_src,
                feat_tgt,
                labels=labels_for_contrast,
            )
            if translation_enabled
            else None
        )

        if (
            translation_enabled
            and adversarial_scale > 0
            and step % args.discriminator_update_interval == 0
        ):
            models.dual_adapter.set_discriminators_trainable(True)
            optimizer_disc.zero_grad(set_to_none=True)
            loss_dual_d, d_logs = models.dual_adapter.compute_discriminator_loss(dual_outputs)
            if not bool(torch.isfinite(loss_dual_d)):
                raise FloatingPointError(
                    f"Non-finite discriminator loss at epoch={epoch}, step={step}."
                )
            loss_dual_d.backward()
            discriminator_parameters = list(models.dual_adapter.discriminator_parameters())
            if args.grad_clip > 0:
                grad_norm_disc = torch.nn.utils.clip_grad_norm_(
                    discriminator_parameters,
                    args.grad_clip,
                )
                grad_norm_disc = float(grad_norm_disc.detach().cpu())
            else:
                grad_norm_disc = _gradient_norm(discriminator_parameters)
            optimizer_disc.step()
            totals["loss_dual_d"] += float(loss_dual_d.detach().cpu())
            totals["grad_norm_disc"] += grad_norm_disc
            if args.grad_clip > 0 and grad_norm_disc > args.grad_clip:
                totals["grad_clip_disc_steps"] += 1.0
            _accumulate_logs(totals, d_logs)
            totals["disc_steps"] += 1.0

        if translation_enabled:
            models.dual_adapter.set_discriminators_trainable(False)
        optimizer_main.zero_grad(set_to_none=True)

        pred_src = models.classifier(feat_src)
        pred_tgt = models.classifier(feat_tgt)
        loss_cls_source = criterion_cls(pred_src, source_labels)
        loss_cls_target = criterion_cls(pred_tgt, target_labels)
        target_cls_weight = max(
            float(getattr(args, "target_classification_weight", 1.0)),
            0.0,
        )
        loss_cls = loss_cls_source + target_cls_weight * loss_cls_target

        if translation_enabled:
            loss_dual_g, g_logs = models.dual_adapter.compute_generator_loss(
                outputs=dual_outputs,
                labels=labels_for_contrast,
                classifier=models.classifier,
                criterion_cls=criterion_cls,
                source_labels=source_labels,
                target_labels=target_labels,
                num_classes=num_classes,
                adversarial_scale=adversarial_scale,
                module_c_scale=module_c_scale,
                modality_drift_scale=modality_drift_scale,
                module_c_enabled=bool(getattr(args, "module_c_enabled", True)),
                modality_drift_enabled=bool(
                    getattr(args, "modality_drift_enabled", True)
                ),
            )
            _accumulate_logs(totals, g_logs)
        else:
            loss_dual_g = feat_src.new_zeros(())
        loss_total = loss_cls + args.tal_weight * loss_tal + loss_dual_g
        if not bool(torch.isfinite(loss_total)):
            raise FloatingPointError(
                f"Non-finite main loss at epoch={epoch}, step={step}."
            )
        loss_total.backward()

        ais_parameters = (
            [p for p in models.net_ais.parameters() if p.requires_grad]
            if models.net_ais is not None
            else []
        )
        main_parameters = [
            *[p for p in models.net_vis.parameters() if p.requires_grad],
            *[p for p in models.net_ir.parameters() if p.requires_grad],
            *ais_parameters,
            *[p for p in models.tal.parameters() if p.requires_grad],
            *[
                p
                for p in models.dual_adapter.generator_parameters()
                if p.requires_grad
            ],
            *list(models.classifier.parameters()),
        ]
        if args.grad_clip > 0:
            grad_norm_main = torch.nn.utils.clip_grad_norm_(
                main_parameters,
                args.grad_clip,
            )
            grad_norm_main = float(grad_norm_main.detach().cpu())
        else:
            grad_norm_main = _gradient_norm(main_parameters)
        if args.grad_clip > 0 and grad_norm_main > args.grad_clip:
            totals["grad_clip_main_steps"] += 1.0

        optimizer_main.step()
        models.tal.apply_orthogonal_projection()
        if translation_enabled:
            models.dual_adapter.set_discriminators_trainable(True)

        with torch.no_grad():
            pred_source_labels = torch.argmax(pred_src.detach(), dim=1)
            pred_target_labels = torch.argmax(pred_tgt.detach(), dim=1)
            source_like_features = (
                dual_outputs.source_like.detach() if translation_enabled else feat_tgt.detach()
            )
            target_like_features = (
                dual_outputs.target_like.detach() if translation_enabled else feat_src.detach()
            )
            source_like_logits = models.classifier(source_like_features)
            target_like_logits = models.classifier(target_like_features)
            pred_source_like_labels = torch.argmax(source_like_logits, dim=1)
            pred_target_like_labels = torch.argmax(target_like_logits, dim=1)

        totals["source_correct"] += float((pred_source_labels == source_labels).sum().item())
        totals["target_correct"] += float((pred_target_labels == target_labels).sum().item())
        totals["source_like_correct"] += float(
            (pred_source_like_labels == target_labels).sum().item()
        )
        totals["target_like_correct"] += float(
            (pred_target_like_labels == source_labels).sum().item()
        )
        totals["sample_total"] += float(target_labels.numel())
        totals["loss_total"] += float(loss_total.detach().cpu())
        totals["loss_cls"] += float(loss_cls.detach().cpu())
        totals["loss_cls_source"] += float(loss_cls_source.detach().cpu())
        totals["loss_cls_target"] += float(loss_cls_target.detach().cpu())
        totals["loss_tal"] += float(loss_tal.detach().cpu())
        totals["loss_dual_g"] += float(loss_dual_g.detach().cpu())
        totals["grad_norm_main"] += grad_norm_main
        totals["steps"] += 1.0

    steps = max(totals["steps"], 1.0)
    disc_steps = max(totals["disc_steps"], 1.0)
    sample_total = max(totals["sample_total"], 1.0)
    return {
        **_training_component_status(args),
        "epoch": epoch,
        "train_loss": totals["loss_total"] / steps,
        "train_loss_cls": totals["loss_cls"] / steps,
        "train_loss_cls_source": totals["loss_cls_source"] / steps,
        "train_loss_cls_target": totals["loss_cls_target"] / steps,
        "train_loss_tal": totals["loss_tal"] / steps,
        "train_loss_dual_g": totals["loss_dual_g"] / steps,
        "train_loss_dual_d": totals["loss_dual_d"] / disc_steps,
        "train_grad_norm_main": totals["grad_norm_main"] / steps,
        "train_grad_norm_discriminator": totals["grad_norm_disc"] / disc_steps,
        "train_grad_clip_fraction_main": totals["grad_clip_main_steps"] / steps,
        "train_grad_clip_fraction_discriminator": (
            totals["grad_clip_disc_steps"] / disc_steps
        ),
        "train_discriminator_steps": totals["disc_steps"],
        "train_adversarial_scale": adversarial_scale,
        "train_module_c_scale": module_c_scale,
        "train_acc": totals["target_correct"] / sample_total,
        "train_acc_source": totals["source_correct"] / sample_total,
        "train_acc_target": totals["target_correct"] / sample_total,
        "train_acc_source_like": totals["source_like_correct"] / sample_total,
        "train_acc_target_like": totals["target_like_correct"] / sample_total,
        "train_dual_d_discriminator_total": _average_logged_metric(
            totals,
            "dual_d_discriminator_total",
            disc_steps,
        ),
        "train_dual_d_discriminator_primary": _average_logged_metric(
            totals,
            "dual_d_discriminator_primary",
            disc_steps,
        ),
        "train_dual_d_discriminator_auxiliary": _average_logged_metric(
            totals,
            "dual_d_discriminator_auxiliary",
            disc_steps,
        ),
        "train_dual_d_generator_total": _average_logged_metric(
            totals,
            "dual_d_generator_total",
            steps,
        ),
        "train_dual_d_adv_primary": _average_logged_metric(
            totals,
            "dual_d_adv_primary",
            steps,
        ),
        "train_dual_d_adv_auxiliary": _average_logged_metric(
            totals,
            "dual_d_adv_auxiliary",
            steps,
        ),
        "train_dual_d_cycle": _average_logged_metric(totals, "dual_d_cycle", steps),
        "train_dual_d_identity": _average_logged_metric(totals, "dual_d_identity", steps),
        "train_dual_d_contrastive": _average_logged_metric(
            totals,
            "dual_d_contrastive",
            steps,
        ),
        "train_dual_d_prototype_contrastive": _average_logged_metric(
            totals,
            "dual_d_prototype_contrastive",
            steps,
        ),
        "train_dual_d_classification_feedback": _average_logged_metric(
            totals,
            "dual_d_classification_feedback",
            steps,
        ),
        "train_dual_d_modality_drift": _average_logged_metric(
            totals,
            "dual_d_modality_drift",
            steps,
        ),
        "train_dual_d_modality_alignment_source_before": _average_logged_metric(
            totals,
            "dual_d_modality_alignment_source_before",
            steps,
        ),
        "train_dual_d_modality_alignment_target_like_after": _average_logged_metric(
            totals,
            "dual_d_modality_alignment_target_like_after",
            steps,
        ),
        "train_dual_d_modality_drift_source_to_target_raw": _average_logged_metric(
            totals,
            "dual_d_modality_drift_source_to_target_raw",
            steps,
        ),
        "train_dual_d_modality_alignment_target_before": _average_logged_metric(
            totals,
            "dual_d_modality_alignment_target_before",
            steps,
        ),
        "train_dual_d_modality_alignment_source_like_after": _average_logged_metric(
            totals,
            "dual_d_modality_alignment_source_like_after",
            steps,
        ),
        "train_dual_d_modality_drift_target_to_source_raw": _average_logged_metric(
            totals,
            "dual_d_modality_drift_target_to_source_raw",
            steps,
        ),
        "train_dual_d_weighted_cycle": _average_logged_metric(
            totals, "dual_d_weighted_cycle", steps
        ),
        "train_dual_d_weighted_identity": _average_logged_metric(
            totals, "dual_d_weighted_identity", steps
        ),
        "train_dual_d_weighted_contrastive": _average_logged_metric(
            totals, "dual_d_weighted_contrastive", steps
        ),
        "train_dual_d_weighted_prototype_contrastive": _average_logged_metric(
            totals, "dual_d_weighted_prototype_contrastive", steps
        ),
        "train_dual_d_weighted_classification_feedback": _average_logged_metric(
            totals, "dual_d_weighted_classification_feedback", steps
        ),
        "train_modality_drift_scale": modality_drift_scale,
        "train_dual_d_weighted_modality_drift": _average_logged_metric(
            totals, "dual_d_weighted_modality_drift", steps
        ),
    }


@torch.no_grad()
def evaluate(
    args,
    models: ModelBundle,
    dataloader: DataLoader,
    criterion_cls: nn.Module,
    device: torch.device,
    num_classes: int,
    feature_mode: str | None = None,
    domain: str = "target",
) -> Dict[str, object]:
    """Evaluate one explicitly named domain without crossing inference paths."""

    domain = str(domain).lower()
    if domain not in {"source", "target"}:
        raise ValueError("Evaluation domain must be 'source' or 'target'.")

    models.net_vis.eval()
    models.net_ir.eval()
    if models.net_ais is not None:
        models.net_ais.eval()
    models.tal.eval()
    models.dual_adapter.eval()
    models.classifier.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        labels = batch["label"].to(device, non_blocking=True)
        model_mode = _model_mode(args)
        selected_mode = feature_mode or args.eval_feature_mode
        if model_mode == "dual_d":
            modalities = _encode_batch_modalities(models, batch, device)
            projected = (
                models.tal.project_source(modalities)
                if domain == "source"
                else models.tal.project_target(modalities)
            )
            features = torch.cat(projected, dim=1)
            if domain == "target" and selected_mode != "raw":
                features = models.dual_adapter.inference_features(
                    features,
                    mode=selected_mode,
                )
            elif domain == "source":
                selected_mode = "raw"
        else:
            features = _encode_baseline_features(
                models,
                batch,
                device,
                model_mode,
            )
            selected_mode = "raw"
        logits = models.classifier(features)
        loss = criterion_cls(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        total_loss += float(loss.detach().cpu())
        steps += 1
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    predictions_tensor = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    labels_tensor = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    metrics = classification_metrics(predictions_tensor, labels_tensor, num_classes)
    metrics["val_loss"] = total_loss / max(steps, 1)
    metrics["feature_mode"] = selected_mode
    return metrics


def _balanced_snapshot_indices(labels: List[int], max_samples: int) -> List[int]:
    """Select a deterministic class-balanced subset without replacement."""

    if len(labels) <= max_samples:
        return list(range(len(labels)))
    buckets: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(int(label), []).append(index)
    selected: List[int] = []
    offset = 0
    class_ids = sorted(buckets)
    while len(selected) < max_samples:
        progressed = False
        for class_id in class_ids:
            candidates = buckets[class_id]
            if offset < len(candidates):
                selected.append(candidates[offset])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        offset += 1
    return selected


@torch.no_grad()
def save_feature_embeddings(
    models: ModelBundle,
    source_dataloader: DataLoader,
    target_dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_samples: int = 512,
    translation_enabled: bool = True,
) -> None:
    """Save source and target features for post-hoc alignment diagnostics.

    This snapshot is intentionally small and contains no model weights. It is
    written only when the monitored validation metric improves, so ablation
    runs can visualize representations without producing large checkpoints.
    Dimensionality reduction is deliberately not performed here or anywhere in
    the train/inference path; any 2-D projection belongs only to later plots.
    """

    max_samples = max(int(max_samples), 1)
    models.net_vis.eval()
    models.net_ir.eval()
    if models.net_ais is not None:
        models.net_ais.eval()
    models.tal.eval()
    models.dual_adapter.eval()

    def collect(dataloader: DataLoader, domain: str):
        dataset_labels = getattr(dataloader.dataset, "labels", None)
        if dataset_labels is not None and len(dataset_labels) > max_samples:
            selected_indices = _balanced_snapshot_indices(
                list(dataset_labels), max_samples
            )
            dataloader = DataLoader(
                Subset(dataloader.dataset, selected_indices),
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=min(int(dataloader.num_workers), 4),
                pin_memory=bool(dataloader.pin_memory),
                collate_fn=dataloader.collate_fn,
            )
        raw_parts = []
        pre_modality_parts: List[List[np.ndarray]] = []
        post_modality_parts: List[List[np.ndarray]] = []
        generated_parts: Dict[str, List[np.ndarray]] = {}
        label_parts = []
        sample_ids = []
        sample_count = 0
        for batch in dataloader:
            modalities = _encode_batch_modalities(models, batch, device)
            projected = (
                models.tal.project_source(modalities)
                if domain == "source"
                else models.tal.project_target(modalities)
            )
            raw = torch.cat(projected, dim=1)
            remaining = max_samples - sample_count
            raw_parts.append(raw[:remaining].cpu().numpy())
            while len(pre_modality_parts) < len(modalities):
                pre_modality_parts.append([])
                post_modality_parts.append([])
            for modality_index, (before, after) in enumerate(zip(modalities, projected)):
                pre_modality_parts[modality_index].append(
                    before[:remaining].cpu().numpy()
                )
                post_modality_parts[modality_index].append(
                    after[:remaining].cpu().numpy()
                )
            generated = {"raw_logits": models.classifier(raw)}
            if translation_enabled:
                translator = models.dual_adapter.coordinator.translator
                if domain == "source":
                    target_like, reconstruction = translator.cycle_from_source(raw)
                    generated.update(
                        {
                            "target_like": target_like,
                            "reconstruction": reconstruction,
                            "identity": translator.target_to_source(raw),
                            "target_like_logits": models.classifier(target_like),
                        }
                    )
                else:
                    source_like, reconstruction = translator.cycle_from_target(raw)
                    generated.update(
                        {
                            "source_like": source_like,
                            "reconstruction": reconstruction,
                            "identity": translator.source_to_target(raw),
                            "source_like_logits": models.classifier(source_like),
                        }
                    )
            for name, features in generated.items():
                generated_parts.setdefault(name, []).append(
                    features[:remaining].cpu().numpy()
                )
            label_parts.append(batch["label"][:remaining].cpu().numpy())
            if "sample_id" in batch:
                sample_ids.extend(list(batch["sample_id"])[:remaining])
            else:
                vis_paths = list(batch.get("vis_path", []))[:remaining]
                ir_paths = list(batch.get("ir_path", []))[:remaining]
                sample_ids.extend(
                    f"{vis_path}|{ir_path}"
                    for vis_path, ir_path in zip(vis_paths, ir_paths)
                )
            sample_count += min(int(raw.size(0)), remaining)
            if sample_count >= max_samples:
                break
        raw_array = (
            np.concatenate(raw_parts, axis=0)
            if raw_parts
            else np.empty((0, 0), dtype=np.float32)
        )
        generated_arrays = {
            name: np.concatenate(parts, axis=0)
            for name, parts in generated_parts.items()
        }
        label_array = (
            np.concatenate(label_parts, axis=0).astype(np.int64)
            if label_parts
            else np.empty(0, dtype=np.int64)
        )
        pre_arrays = [np.concatenate(parts, axis=0) for parts in pre_modality_parts]
        post_arrays = [np.concatenate(parts, axis=0) for parts in post_modality_parts]
        return (
            raw_array,
            generated_arrays,
            label_array,
            np.asarray(sample_ids, dtype=str),
            pre_arrays,
            post_arrays,
        )

    (
        source_raw,
        source_generated,
        source_labels,
        source_ids,
        source_pre,
        source_post,
    ) = collect(source_dataloader, "source")
    (
        target_raw,
        target_generated,
        target_labels,
        target_ids,
        target_pre,
        target_post,
    ) = collect(target_dataloader, "target")
    if not len(source_labels) or not len(target_labels):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "source_raw": source_raw,
        "source_raw_logits": source_generated["raw_logits"],
        "source_labels": source_labels,
        "source_sample_ids": source_ids,
        "target_raw": target_raw,
        "target_raw_logits": target_generated["raw_logits"],
        "target_labels": target_labels,
        "target_sample_ids": target_ids,
        "raw": target_raw,
        "labels": target_labels,
    }
    modality_names = ["sar", "optical"] + [
        f"modality_{index}" for index in range(2, len(source_pre))
    ]
    for index, name in enumerate(modality_names[: len(source_pre)]):
        arrays[f"source_{name}_pre_alignment"] = source_pre[index]
        arrays[f"source_{name}_post_alignment"] = source_post[index]
        arrays[f"target_{name}_pre_alignment"] = target_pre[index]
        arrays[f"target_{name}_post_alignment"] = target_post[index]
    if translation_enabled:
        arrays.update(
            {
                "source_target_like": source_generated["target_like"],
                "source_reconstruction": source_generated["reconstruction"],
                "source_identity": source_generated["identity"],
                "source_target_like_logits": source_generated["target_like_logits"],
                "target_source_like": target_generated["source_like"],
                "target_reconstruction": target_generated["reconstruction"],
                "target_identity": target_generated["identity"],
                "target_source_like_logits": target_generated["source_like_logits"],
                # Backward-compatible alias for older analysis consumers.
                "source_like": target_generated["source_like"],
            }
        )
    np.savez_compressed(output_path, **arrays)


def checkpoint_state(
    args,
    models: ModelBundle,
    optimizer_main,
    optimizer_disc,
    epoch: int,
    metrics: Dict[str, object],
    label_map: Dict[str, int],
) -> Dict[str, object]:
    """Build checkpoint state dictionary."""

    return {
        "epoch": epoch,
        "args": vars(args),
        "label_map": label_map,
        "metrics": metrics,
        "net_vis": models.net_vis.state_dict(),
        "net_ir": models.net_ir.state_dict(),
        "net_ais": models.net_ais.state_dict() if models.net_ais is not None else None,
        "tal": models.tal.state_dict(),
        "dual_adapter": models.dual_adapter.state_dict(),
        "classifier": models.classifier.state_dict(),
        "optimizer_main": optimizer_main.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
    }


def _capture_model_weights(models: ModelBundle) -> Dict[str, Dict[str, torch.Tensor]]:
    """Clone model-only state to CPU for leakage-free final test evaluation."""

    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in models.__dict__.items()
        if module is not None
    }


def _restore_model_weights(
    models: ModelBundle,
    weights: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Restore a model-only snapshot captured by ``_capture_model_weights``."""

    for name, state in weights.items():
        module = getattr(models, name)
        module.load_state_dict(state)


def _artifact_path(run_dir: Path, filename: str, args) -> Path:
    """Return an iteration-safe artifact path.

    Grouped matrix runs share one directory, so every per-iteration artifact
    receives a stable suffix while retaining the legacy names for standalone
    runs.
    """

    suffix = str(getattr(args, "artifact_suffix", "") or "").strip()
    path = Path(filename)
    if suffix:
        return run_dir / f"{path.stem}_{suffix}{path.suffix}"
    return run_dir / path


def _checkpoint_path(run_dir: Path, filename: str, args) -> Path:
    """Return a checkpoint path that cannot be overwritten by another iteration."""

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return _artifact_path(checkpoint_dir, filename, args)


def run_training(args) -> Dict[str, object]:
    """Run a complete standalone Dual_D training experiment."""

    set_seed(
        args.seed,
        deterministic=bool(getattr(args, "deterministic_training", False)),
    )
    device = resolve_device(args.device)
    validate_cuda_architecture(device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = getattr(args, "dataset_type", "directory")
    default_target_name = (
        "m4sar_target"
        if dataset_type == "m4sar_classification"
        else Path(args.target_root).name
    )
    run_name = args.run_name or f"dual_d_{default_target_name}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_text_logger(_artifact_path(run_dir, "train.log", args))
    metrics_logger = CSVMetricLogger(_artifact_path(run_dir, "metrics.csv", args))
    per_class_logger = CSVMetricLogger(
        _artifact_path(run_dir, "per_class_metrics.csv", args)
    )

    logger.info("Starting Dual_D standalone training")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")

    (
        source_train,
        source_train_eval,
        target_train,
        target_train_eval,
        target_val,
        target_test,
        label_map,
    ) = build_datasets(args)
    if bool(getattr(args, "use_ais", False)):
        ais_lengths = {
            int(dataset.ais_signal_length)
            for dataset in (
                source_train,
                source_train_eval,
                target_train,
                target_train_eval,
                target_val,
                *([target_test] if target_test is not None else []),
            )
        }
        if len(ais_lengths) != 1:
            raise RuntimeError(
                "AIS signal lengths differ across source/target splits: "
                f"{sorted(ais_lengths)}. Use one MAT file or a common --ais-sequence-length."
            )
        args.effective_ais_sequence_length = ais_lengths.pop()
        logger.info(
            "AIS input: JMDA-compatible I/Q tensor shape [2, %d]",
            args.effective_ais_sequence_length,
        )
        args.effective_ais_alignment = getattr(
            target_train,
            "ais_alignment_mode",
            "per_sample",
        )
    num_classes = len(label_map)
    if _model_mode(args) != "dual_d" or not bool(
        getattr(args, "translation_enabled", True)
    ):
        args.eval_feature_mode = "raw"
    criterion_cls = build_classification_criterion(
        args,
        source_train,
        device,
        num_classes,
    )
    save_json(
        {"args": vars(args), "label_map": label_map},
        _artifact_path(run_dir, "resolved_config.json", args),
    )
    save_json(label_map, _artifact_path(run_dir, "label_map.json", args))

    logger.info(f"Source train samples: {len(source_train)}")
    logger.info(f"Target train samples: {len(target_train)}")
    logger.info(f"Target adaptation val samples: {len(target_val)}")
    if target_test is not None:
        logger.info(f"Target final test samples: {len(target_test)}")
    logger.info(f"Classes: {num_classes}")

    is_so2sat = dataset_type == "so2sat_lcz42"
    is_m4sar = dataset_type == "m4sar_classification"
    if is_so2sat:
        if target_test is None:
            raise RuntimeError("So2Sat requires an independent target-test split.")
        data_audit = _audit_so2sat_protocol(
            source_train,
            target_train,
            target_val,
            target_test,
        )
        audit_errors = []
        if data_audit["leakage_detected"]:
            audit_errors.append("So2Sat target adaptation/test split leakage detected")
        if not data_audit["closed_set_17_classes"]:
            audit_errors.append("So2Sat splits do not share the complete 17-class set")
        logger.info(
            "So2Sat audit: target-adapt train/val=%d/%d | test=%d | "
            "adapt_row_overlap=%d | separate_test=%s | closed_set_17=%s",
            len(target_train),
            len(target_val),
            len(target_test),
            data_audit["target_adapt_row_overlap_count"],
            data_audit["target_test_is_separate_file"],
            data_audit["closed_set_17_classes"],
        )
    elif is_m4sar:
        data_audit = _audit_m4sar_protocol(
            source_train,
            target_train,
            target_val,
        )
        audit_errors = []
        if data_audit["leakage_detected"]:
            audit_errors.append("M4-SAR manifest domain/split protocol is invalid")
        logger.info(
            "M4-SAR audit: Source/Target train=%d/%d | Target val=%d | "
            "paired metadata retained=%s | paired sampling=%s | Target test opened=%s",
            len(source_train),
            len(target_train),
            len(target_val),
            data_audit["physical_pair_metadata_retained"],
            data_audit["pair_correspondence_used_for_training"],
            data_audit["target_test_opened_during_selection"],
        )
    else:
        data_audit = audit_dataset_splits(
            target_train,
            target_val,
            hash_contents=bool(getattr(args, "data_audit_hashes", False)),
        )
        audit_errors = data_audit_errors(data_audit)
        logger.info(
            "Data audit: same_dir=%s | path_overlap(vis/ir/ais)=%d/%d/%d | "
            "content_overlap(vis/ir/ais)=%d/%d/%d | ais_index_overlap=%d | "
            "stem_mismatch(vis-ir/vis-ais)=%d/%d",
            data_audit["same_base_dir"],
            data_audit["path_overlap_vis_count"],
            data_audit["path_overlap_ir_count"],
            data_audit["path_overlap_ais_count"],
            data_audit["content_overlap_vis_count"],
            data_audit["content_overlap_ir_count"],
            data_audit["content_overlap_ais_count"],
            data_audit.get("ais_index_overlap_count", 0),
            data_audit["vis_ir_stem_mismatch_count"],
            data_audit["vis_ais_stem_mismatch_count"],
        )
    save_json(data_audit, _artifact_path(run_dir, "data_audit.json", args))
    if audit_errors:
        message = "Data audit failed: " + "; ".join(audit_errors)
        if bool(getattr(args, "strict_data_audit", True)):
            raise RuntimeError(message)
        logger.warning(message)

    class_summaries = {
        "source_train": summarize_label_distribution(source_train, num_classes, label_map),
        "target_train": summarize_label_distribution(target_train, num_classes, label_map),
        "target_adapt_val": summarize_label_distribution(target_val, num_classes, label_map),
    }
    if target_test is not None:
        class_summaries["target_test"] = summarize_label_distribution(
            target_test, num_classes, label_map
        )
    for split_name, split_summary in class_summaries.items():
        logger.info(
            "%s classes: present %d/%d | absent %d | counts [%s]",
            split_name,
            split_summary["present_class_count"],
            split_summary["num_classes"],
            split_summary["absent_class_count"],
            _compact_distribution(split_summary),
        )

    adaptive_batch_enabled = bool(
        is_m4sar and getattr(args, "adaptive_batch_size", False)
    )
    adaptive_batch_plan = None
    adaptive_memory_gb = None
    configured_batch_size = int(args.batch_size)
    args.configured_batch_size = configured_batch_size
    if adaptive_batch_enabled:
        if device.type != "cuda":
            raise ValueError("Adaptive batch sizing requires a CUDA device.")
        adaptive_batch_plan = parse_adaptive_batch_plan(
            getattr(args, "adaptive_batch_plan", "10:32,18:64,26:128")
        )
        adaptive_memory_gb = reusable_cuda_memory_gb(device)
        args.batch_size = select_adaptive_batch_size(
            adaptive_batch_plan,
            adaptive_memory_gb,
        )
        args.effective_initial_batch_size = int(args.batch_size)
        logger.info(
            "Adaptive batch sizing enabled: reusable_memory=%.2f GiB | "
            "configured_batch=%d | selected_batch=%d | plan=%s",
            adaptive_memory_gb,
            configured_batch_size,
            args.batch_size,
            getattr(args, "adaptive_batch_plan", ""),
        )

    if is_m4sar:
        args.pin_memory = device.type == "cuda"
        paired_loader = IndependentDomainLoaders(source_train, target_train, args)
        paired_common_classes = sorted(
            set(int(value) for value in source_train.labels)
            & set(int(value) for value in target_train.labels)
        )
    else:
        paired_loader = PairedClassSampler(
            source_train,
            target_train,
            args.batch_size,
            min_steps_per_epoch=getattr(args, "min_steps_per_epoch", 4),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        paired_common_classes = list(paired_loader.classes)
    save_json(
        {
            "label_map": label_map,
            "paired_training_classes": paired_common_classes,
            "source_target_loader_policy": (
                "independent_shuffle_no_pair_correspondence"
                if is_m4sar
                else "class_matched_sampler"
            ),
            **class_summaries,
        },
        _artifact_path(run_dir, "class_distribution.json", args),
    )
    logger.info(
        "Training classes shared by domains: %d/%d | class ids [%s]",
        len(paired_common_classes),
        num_classes,
        ", ".join(str(class_id) for class_id in paired_common_classes),
    )
    logger.info(
        "Runtime profile: batch=%d | training_steps=%d | workers=%d | "
        "train_eval_every=%d | raw_eval_every=%d | stability_window=%d | "
        "lr_scheduler_start=%d | checkpoint_select_from=%d | "
        "early_stop=%d after min_epoch=%d",
        args.batch_size,
        len(paired_loader),
        args.num_workers,
        max(int(getattr(args, "train_eval_interval", 1)), 1),
        max(int(getattr(args, "raw_eval_interval", 5)), 1),
        max(int(getattr(args, "monitor_stability_window", 1)), 1),
        max(int(getattr(args, "lr_scheduler_start_epoch", 1)), 1),
        max(int(getattr(args, "checkpoint_selection_min_epoch", 1)), 1),
        int(getattr(args, "early_stopping_patience", 0)),
        int(getattr(args, "early_stopping_min_epochs", 0)),
    )
    base_augmentation = float(getattr(args, "augmentation_strength", 0.0))
    visible_augmentation = getattr(args, "vis_augmentation_strength", None)
    infrared_augmentation = getattr(args, "ir_augmentation_strength", None)
    if is_m4sar:
        logger.info(
            "M4-SAR modalities: Optical RGB (3ch) + SAR grayscale (1ch) | "
            "TAL block order=SAR,Optical | loader policy=independent shuffle"
        )
        logger.info(
            "M4-SAR model mode: %s | alignment=%s | translation=%s | "
            "Module-C=%s | Relation-Drift=%s | Target test is deferred until after "
            "validation checkpoint selection",
            _model_mode(args),
            getattr(args, "alignment_mode", "tal"),
            bool(getattr(args, "translation_enabled", True)),
            bool(getattr(args, "module_c_enabled", True)),
            bool(getattr(args, "modality_drift_enabled", True)),
        )
        if _model_mode(args) != "dual_d":
            logger.warning(
                "SOURCE-ONLY BASELINE ACTIVE: TAL, translators, discriminators, "
                "Module C, and Relation Drift are disabled; their metrics are n/a. "
                "train_acc is Source-domain online accuracy, while validation is "
                "Target-domain accuracy. Use --model-mode dual_d for the full method."
            )
    elif is_so2sat:
        logger.info(
            "Satellite augmentation: synchronized S1/S2 geometry=%s | "
            "SAR clip after Z-score=%s",
            bool(getattr(args, "satellite_augment", True)),
            getattr(args, "sar_clip_after_normalize", None),
        )
        logger.info(
            "Satellite modalities: registered S1 SAR (8ch) + S2 Optical (10ch) | "
            "TAL block order=SAR,Optical"
        )
    else:
        logger.info(
            "Augmentation profile: base=%.3f | visible=%.3f | infrared=%.3f | "
            "freeze_frozen_batch_norm_stats=%s",
            base_augmentation,
            base_augmentation if visible_augmentation is None else float(visible_augmentation),
            base_augmentation if infrared_augmentation is None else float(infrared_augmentation),
            bool(getattr(args, "freeze_frozen_batch_norm_stats", False)),
        )
        logger.info(
            "VIS/IR pairing: %s | count-mismatch classes train/val: %d/%d | "
            "AIS alignment: %s | AIS rows train/val: %s/%s",
            "synchronized" if getattr(args, "synchronize_modalities", False) else "class-level-unpaired",
            getattr(source_train, "vis_ir_count_mismatch_count", 0),
            getattr(target_val, "vis_ir_count_mismatch_count", 0),
            getattr(source_train, "ais_alignment_mode", "none"),
            getattr(source_train, "reference_ais_pool_indices", np.empty(0)).size,
            getattr(target_val, "reference_ais_pool_indices", np.empty(0)).size,
        )
    eval_batch_size = (
        min(batch_size for _threshold, batch_size in adaptive_batch_plan)
        if adaptive_batch_plan is not None
        else args.batch_size
    )
    val_loader = DataLoader(
        target_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    test_loader = (
        DataLoader(
            target_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        if target_test is not None and bool(getattr(args, "evaluate_target_test", True))
        else None
    )
    if target_test is not None and test_loader is None:
        logger.info(
            "Final target-test evaluation is disabled for this run; testing.h5 samples "
            "will not be loaded."
        )
    source_eval_loader = DataLoader(
        source_train_eval,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        target_train_eval,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    models = build_models(args, num_classes, device)
    component_status = _training_component_status(args)
    save_json(
        component_status,
        _artifact_path(run_dir, "training_component_status.json", args),
    )
    args.effective_dual_config = models.dual_adapter.config.to_dict()
    save_json(
        {"args": vars(args), "label_map": label_map},
        _artifact_path(run_dir, "resolved_config.json", args),
    )
    save_json(
        args.effective_dual_config,
        _artifact_path(run_dir, "resolved_dual_config.json", args),
    )
    if component_status["translator_active"]:
        logger.info(
            "Effective Dual-D loss weights: %s",
            json.dumps(args.effective_dual_config["loss_weights"], sort_keys=True),
        )
        logger.info(
            "Modality drift: dims=%s | margin=%.6f | warmup=%d | ramp=%d",
            args.effective_dual_config.get("modality_dims", []),
            float(args.effective_dual_config.get("modality_drift_margin", 0.0)),
            max(int(getattr(args, "modality_drift_warmup_epochs", 0)), 0),
            max(int(getattr(args, "modality_drift_ramp_epochs", 0)), 0),
        )
    elif component_status["model_mode"] == "dual_d":
        logger.info(
            "Active components: encoders + %s alignment + classifier | inactive: "
            "translators, discriminators, Module C, Relation Drift",
            component_status["alignment_mode"],
        )
    else:
        logger.info(
            "Active components: baseline encoder(s) + classifier | inactive: "
            "TAL, translators, discriminators, Module C, Relation Drift"
        )
    multi_gpu_active = isinstance(models.net_vis, nn.DataParallel)
    if bool(getattr(args, "multi_gpu", False)) and device.type == "cuda":
        logger.info(
            "CUDA devices available: %d | encoder data parallel active=%s",
            torch.cuda.device_count(),
            multi_gpu_active,
        )
    active_models = [model for model in models.__dict__.values() if model is not None]
    total_parameters = sum(
        parameter.numel() for model in active_models for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for model in active_models
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "Model parameters: total=%d | trainable=%d",
        total_parameters,
        trainable_parameters,
    )
    optimizer_main, optimizer_disc = build_optimizers(args, models)
    monitor_metric = getattr(args, "monitor_metric", "val_acc")
    monitor_mode = "min" if monitor_metric == "val_loss" else "max"
    scheduler_main = ReduceLROnPlateau(
        optimizer_main,
        mode=monitor_mode,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    scheduler_disc = ReduceLROnPlateau(
        optimizer_disc,
        mode=monitor_mode,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=getattr(args, "min_lr_discriminator", args.min_lr),
    )

    best_acc = -1.0
    best_acc_epoch = 0
    best_f1 = -1.0
    best_f1_epoch = 0
    best_selection_score = float("-inf")
    best_metrics: Dict[str, object] = {}
    monitor_history: List[float] = []
    start_time = time.time()
    epochs_without_improvement = 0
    epochs_completed = 0
    early_stopped = False
    best_model_weights = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if adaptive_batch_enabled:
            minimum_memory_gb = max(
                float(getattr(args, "adaptive_min_free_gb", 10.0)),
                float(adaptive_batch_plan[0][0]),
            )
            adaptive_memory_gb = reusable_cuda_memory_gb(device)
            poll_seconds = max(
                float(getattr(args, "adaptive_memory_poll_seconds", 60.0)),
                1.0,
            )
            while adaptive_memory_gb < minimum_memory_gb:
                logger.info(
                    "Epoch %d paused: reusable CUDA memory %.2f GiB is below "
                    "the %.2f GiB safety gate; rechecking in %.0fs.",
                    epoch,
                    adaptive_memory_gb,
                    minimum_memory_gb,
                    poll_seconds,
                )
                torch.cuda.empty_cache()
                time.sleep(poll_seconds)
                adaptive_memory_gb = reusable_cuda_memory_gb(device)
            selected_batch_size = select_adaptive_batch_size(
                adaptive_batch_plan,
                adaptive_memory_gb,
            )
            if selected_batch_size != int(args.batch_size):
                previous_batch_size = int(args.batch_size)
                args.batch_size = selected_batch_size
                paired_loader = IndependentDomainLoaders(
                    source_train,
                    target_train,
                    args,
                    seed=int(args.seed) + epoch * 10_007,
                )
                logger.info(
                    "Epoch %d adaptive batch change: %d -> %d at %.2f GiB "
                    "reusable CUDA memory.",
                    epoch,
                    previous_batch_size,
                    selected_batch_size,
                    adaptive_memory_gb,
                )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_phase_start = time.time()
        train_metrics = train_one_epoch(
            args=args,
            models=models,
            paired_loader=paired_loader,
            optimizer_main=optimizer_main,
            optimizer_disc=optimizer_disc,
            criterion_cls=criterion_cls,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
        )
        train_metrics["train_batch_size"] = int(args.batch_size)
        train_metrics["adaptive_reusable_memory_gb"] = adaptive_memory_gb
        train_seconds = time.time() - train_phase_start

        val_phase_start = time.time()
        val_metrics = evaluate(
            args=args,
            models=models,
            dataloader=val_loader,
            criterion_cls=criterion_cls,
            device=device,
            num_classes=num_classes,
        )
        val_seconds = time.time() - val_phase_start

        monitor_values = {
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1_macro_present": float(val_metrics["f1_macro_present"]),
            "val_loss": float(val_metrics["val_loss"]),
        }
        monitor_value = monitor_values[monitor_metric]
        monitor_history.append(monitor_value)
        monitor_stability_window = max(
            int(getattr(args, "monitor_stability_window", 1)), 1
        )
        monitor_stability_window = min(monitor_stability_window, int(args.epochs))
        monitor_selection_score = _stable_monitor_score(
            monitor_history,
            monitor_mode,
            monitor_stability_window,
        )
        min_delta = float(getattr(args, "early_stopping_min_delta", 0.0))
        checkpoint_selection_eligible = _checkpoint_selection_is_eligible(
            args,
            epoch,
        )
        would_improve = (
            checkpoint_selection_eligible
            and monitor_selection_score > best_selection_score + min_delta
        )

        raw_eval_seconds = 0.0
        if args.eval_feature_mode == "raw":
            val_raw_metrics = val_metrics
        else:
            raw_eval_interval = max(int(getattr(args, "raw_eval_interval", 5)), 1)
            if epoch == 1 or epoch % raw_eval_interval == 0 or would_improve:
                raw_eval_start = time.time()
                val_raw_metrics = evaluate(
                    args=args,
                    models=models,
                    dataloader=val_loader,
                    criterion_cls=criterion_cls,
                    device=device,
                    num_classes=num_classes,
                    feature_mode="raw",
                )
                raw_eval_seconds = time.time() - raw_eval_start
            else:
                val_raw_metrics = {}

        train_eval_interval = max(int(getattr(args, "train_eval_interval", 1)), 1)
        train_eval_seconds = 0.0
        if epoch == 1 or epoch % train_eval_interval == 0:
            train_eval_start = time.time()
            train_full_metrics = evaluate(
                args=args,
                models=models,
                dataloader=train_eval_loader,
                criterion_cls=criterion_cls,
                device=device,
                num_classes=num_classes,
            )
            train_eval_seconds = time.time() - train_eval_start
        else:
            train_full_metrics = {}

        lr_scheduler_active = _lr_scheduler_is_active(args, epoch)
        if lr_scheduler_active:
            scheduler_main.step(monitor_value)
            scheduler_disc.step(monitor_value)

        if args.eval_feature_mode == "raw":
            sampled_value = train_metrics.get("train_acc_target")
        elif args.eval_feature_mode == "source_like":
            sampled_value = train_metrics.get("train_acc_source_like")
        else:
            sampled_value = None
        sampled_mode_acc = (
            float(sampled_value) if sampled_value is not None else float("nan")
        )
        full_train_acc = train_full_metrics.get("accuracy")
        if device.type == "cuda":
            gibibyte = float(1024**3)
            cuda_peak_allocated_gb = (
                torch.cuda.max_memory_allocated(device) / gibibyte
            )
            cuda_peak_reserved_gb = (
                torch.cuda.max_memory_reserved(device) / gibibyte
            )
        else:
            cuda_peak_allocated_gb = 0.0
            cuda_peak_reserved_gb = 0.0

        row = {
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision_macro_present": val_metrics["precision_macro_present"],
            "val_recall_macro_present": val_metrics["recall_macro_present"],
            "val_f1_macro_present": val_metrics["f1_macro_present"],
            "val_precision_micro": val_metrics["precision_micro"],
            "val_recall_micro": val_metrics["recall_micro"],
            "val_f1_micro": val_metrics["f1_micro"],
            "val_raw_loss": val_raw_metrics.get("val_loss"),
            "val_raw_acc": val_raw_metrics.get("accuracy"),
            "val_raw_f1_macro_present": val_raw_metrics.get("f1_macro_present"),
            "train_full_acc": full_train_acc,
            "train_full_f1_macro_present": train_full_metrics.get("f1_macro_present"),
            "target_train_eval_acc": full_train_acc,
            "target_train_eval_f1_macro_present": train_full_metrics.get(
                "f1_macro_present"
            ),
            "train_sampled_minus_full_acc": (
                sampled_mode_acc - float(full_train_acc)
                if full_train_acc is not None and np.isfinite(sampled_mode_acc)
                else None
            ),
            "source_train_minus_target_val_acc": (
                float(train_metrics["train_acc_source"])
                - float(val_metrics["accuracy"])
            ),
            "monitor_value": monitor_value,
            "monitor_selection_score": monitor_selection_score,
            "monitor_stability_window": monitor_stability_window,
            "checkpoint_selection_eligible": checkpoint_selection_eligible,
            "checkpoint_selection_min_epoch": max(
                int(getattr(args, "checkpoint_selection_min_epoch", 1)),
                1,
            ),
            "lr_scheduler_active": lr_scheduler_active,
            "lr_main": optimizer_main.param_groups[-1]["lr"],
            "lr_discriminator": optimizer_disc.param_groups[0]["lr"],
            "lr_discriminator_to_main_ratio": (
                optimizer_disc.param_groups[0]["lr"]
                / max(optimizer_main.param_groups[-1]["lr"], 1e-30)
            ),
            "cuda_peak_allocated_gb": cuda_peak_allocated_gb,
            "cuda_peak_reserved_gb": cuda_peak_reserved_gb,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "raw_eval_seconds": raw_eval_seconds,
            "train_eval_seconds": train_eval_seconds,
            "epoch_seconds": time.time() - epoch_start,
        }
        metrics_logger.write_row(row)

        logger.info(
            "Epoch %03d/%03d | mode %s | loss %.4f | cls %.4f | tal %s | dual_g %s | "
            "dual_d %s | %s_train_acc %.4f | target_train_eval %s | target_val_acc %.4f | "
            "val_f1 %.4f | adv/moduleC/drift %s/%s/%s | "
            "drift(s2t/t2s) %s/%s | disc_steps %.0f | "
            "grad(main/disc) %.3f/%s | lr(main/disc) %.2e/%.2e | "
            "cuda_peak(alloc/resv) %.2f/%.2fGB | "
            "phase(train/val/raw/train_eval) %.1f/%.1f/%.1f/%.1fs | %.1fs",
            epoch,
            args.epochs,
            row["model_mode"],
            row["train_loss"],
            row["train_loss_cls"],
            _format_optional_metric(row["train_loss_tal"]),
            _format_optional_metric(row["train_loss_dual_g"]),
            _format_optional_metric(row["train_loss_dual_d"]),
            row["train_acc_domain"],
            row["train_acc"],
            _format_optional_metric(row["target_train_eval_acc"]),
            row["val_acc"],
            row["val_f1_macro_present"],
            _format_optional_metric(row["train_adversarial_scale"], precision=2),
            _format_optional_metric(row["train_module_c_scale"], precision=2),
            _format_optional_metric(row["train_modality_drift_scale"], precision=2),
            _format_optional_metric(
                row["train_dual_d_modality_drift_source_to_target_raw"]
            ),
            _format_optional_metric(
                row["train_dual_d_modality_drift_target_to_source_raw"]
            ),
            row["train_discriminator_steps"],
            row["train_grad_norm_main"],
            _format_optional_metric(row["train_grad_norm_discriminator"], precision=3),
            row["lr_main"],
            row["lr_discriminator"],
            row["cuda_peak_allocated_gb"],
            row["cuda_peak_reserved_gb"],
            row["train_seconds"],
            row["val_seconds"],
            row["raw_eval_seconds"],
            row["train_eval_seconds"],
            row["epoch_seconds"],
        )

        save_checkpoints = bool(getattr(args, "save_checkpoints", False))
        last_state = None
        if save_checkpoints:
            last_state = checkpoint_state(
                args,
                models,
                optimizer_main,
                optimizer_disc,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                label_map,
            )
            save_checkpoint(last_state, _checkpoint_path(run_dir, "last_model.pt", args))

        current_acc = float(val_metrics["accuracy"])
        current_f1 = float(val_metrics["f1_macro_present"])
        if checkpoint_selection_eligible and current_acc > best_acc:
            best_acc = current_acc
            best_acc_epoch = epoch
            if save_checkpoints and last_state is not None:
                save_checkpoint(
                    last_state,
                    _checkpoint_path(run_dir, "best_acc_model.pt", args),
                )
        if checkpoint_selection_eligible and current_f1 > best_f1:
            best_f1 = current_f1
            best_f1_epoch = epoch
            if save_checkpoints and last_state is not None:
                save_checkpoint(
                    last_state,
                    _checkpoint_path(run_dir, "best_f1_model.pt", args),
                )
        improved = would_improve
        _write_per_class_metrics(
            per_class_logger,
            val_metrics,
            label_map,
            epoch=epoch,
            split="validation",
            domain="target",
            checkpoint_selected=improved,
        )
        if train_full_metrics:
            _write_per_class_metrics(
                per_class_logger,
                train_full_metrics,
                label_map,
                epoch=epoch,
                split="train_eval",
                domain="target",
                checkpoint_selected=False,
            )
        if improved:
            best_selection_score = monitor_selection_score
            epochs_without_improvement = 0
            best_metrics = {
                "train": train_metrics,
                "train_full": train_full_metrics,
                "target_train_eval": train_full_metrics,
                "val": val_metrics,
                "val_raw": val_raw_metrics,
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "monitor_selection_score": monitor_selection_score,
                "monitor_stability_window": monitor_stability_window,
                "checkpoint_selection_min_epoch": max(
                    int(getattr(args, "checkpoint_selection_min_epoch", 1)),
                    1,
                ),
                "epoch": epoch,
            }
            if test_loader is not None or (
                is_m4sar and bool(getattr(args, "evaluate_target_test", True))
            ):
                best_model_weights = _capture_model_weights(models)
            if save_checkpoints and last_state is not None:
                save_checkpoint(last_state, _checkpoint_path(run_dir, "best_model.pt", args))
            save_json(best_metrics, _artifact_path(run_dir, "best_metrics.json", args))
            if (
                bool(getattr(args, "save_feature_embeddings", False))
                and _model_mode(args) == "dual_d"
            ):
                save_feature_embeddings(
                    models=models,
                    source_dataloader=source_eval_loader,
                    target_dataloader=val_loader,
                    device=device,
                    output_path=_artifact_path(run_dir, "feature_embeddings.npz", args),
                    max_samples=getattr(args, "feature_visualization_samples", 512),
                    translation_enabled=bool(
                        getattr(args, "translation_enabled", True)
                    ),
                )
            logger.info(
                "New best %s: raw=%.4f stable_score=%.4f at epoch %d",
                monitor_metric,
                monitor_value,
                monitor_selection_score,
                epoch,
            )
        elif checkpoint_selection_eligible:
            epochs_without_improvement += 1

        epochs_completed = epoch
        early_stopping_patience = int(getattr(args, "early_stopping_patience", 0))
        early_stopping_min_epochs = int(getattr(args, "early_stopping_min_epochs", 0))
        if (
            early_stopping_patience > 0
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            early_stopped = True
            logger.info(
                "Early stopping at epoch %d after %d epochs without %s improvement.",
                epoch,
                epochs_without_improvement,
                monitor_metric,
            )
            break

    if best_metrics:
        _write_per_class_metrics(
            per_class_logger,
            best_metrics["val"],
            label_map,
            epoch=int(best_metrics["epoch"]),
            split="validation_selected",
            domain="target",
            checkpoint_selected=True,
        )

    if is_m4sar and bool(getattr(args, "evaluate_target_test", True)):
        if best_model_weights is None:
            raise RuntimeError(
                "No eligible validation checkpoint was selected; M4-SAR Target test "
                "was not opened."
            )
        source_test = build_m4sar_test_dataset(args, domain="source")
        target_test = build_m4sar_test_dataset(args, domain="target")
        source_test_loader = DataLoader(
            source_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            target_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        logger.info(
            "Checkpoint selection is complete; opening M4-SAR Target test for one "
            "final evaluation (%d samples).",
            len(target_test),
        )

    source_test_metrics = None
    target_test_metrics = None
    if test_loader is not None:
        if best_model_weights is None:
            raise RuntimeError(
                "No eligible checkpoint was selected; final Target test was not run."
            )
        _restore_model_weights(models, best_model_weights)
        if is_m4sar:
            source_test_metrics = evaluate(
                args=args,
                models=models,
                dataloader=source_test_loader,
                criterion_cls=criterion_cls,
                device=device,
                num_classes=num_classes,
                domain="source",
            )
            save_json(
                source_test_metrics,
                _artifact_path(run_dir, "source_test_metrics.json", args),
            )
            _write_per_class_metrics(
                per_class_logger,
                source_test_metrics,
                label_map,
                epoch=int(best_metrics["epoch"]),
                split="test",
                domain="source",
                checkpoint_selected=True,
            )
        target_test_metrics = evaluate(
            args=args,
            models=models,
            dataloader=test_loader,
            criterion_cls=criterion_cls,
            device=device,
            num_classes=num_classes,
        )
        save_json(
            target_test_metrics,
            _artifact_path(run_dir, "target_test_metrics.json", args),
        )
        _write_per_class_metrics(
            per_class_logger,
            target_test_metrics,
            label_map,
            epoch=int(best_metrics["epoch"]),
            split="test",
            domain="target",
            checkpoint_selected=True,
        )
        logger.info(
            "Independent target test at selected epoch %d | ACC %.4f | F1 %.4f",
            int(best_metrics["epoch"]),
            float(target_test_metrics["accuracy"]),
            float(target_test_metrics["f1_macro_present"]),
        )
        if source_test_metrics is not None:
            logger.info(
                "M4-SAR Source test at the same selected checkpoint | ACC %.4f | "
                "F1 %.4f",
                float(source_test_metrics["accuracy"]),
                float(source_test_metrics["f1_macro_present"]),
            )

    summary = {
        "run_dir": str(run_dir),
        "component_status": component_status,
        "best_acc": best_acc,
        "best_acc_epoch": best_acc_epoch,
        "best_f1_macro_present": best_f1,
        "best_f1_epoch": best_f1_epoch,
        "best_monitor_metric": monitor_metric,
        "best_monitor_value": best_metrics.get("monitor_value"),
        "best_monitor_selection_score": best_selection_score,
        "checkpoint_selection_min_epoch": max(
            int(getattr(args, "checkpoint_selection_min_epoch", 1)),
            1,
        ),
        "best_metrics": best_metrics,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "total_seconds": time.time() - start_time,
    }
    result_path = _artifact_path(run_dir, "result_summary.json", args)
    summary["result_path"] = str(result_path)
    save_json(summary, result_path)
    logger.info(
        "Training complete. Peak validation accuracy: %.4f | selected %s: %.4f",
        best_acc,
        monitor_metric,
        float(best_metrics.get("monitor_value", float("nan"))),
    )
    close_text_logger(logger)
    return summary
