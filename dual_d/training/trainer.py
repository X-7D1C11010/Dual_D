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
from pathlib import Path
import random
import time
from typing import Dict, Tuple
import warnings

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dual_d.config import load_config
from dual_d.data import (
    MultiModalDomainDataset,
    PairedClassSampler,
    audit_dataset_splits,
    data_audit_errors,
)
from dual_d.integration_adapter import DualDTrainingAdapter
from dual_d.models import (
    Classifier,
    IRFeatureExtractor,
    LabelSmoothingCrossEntropy,
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

    net_vis: VisualFeatureExtractor
    net_ir: IRFeatureExtractor
    tal: TensorBasedAlignmentStable
    dual_adapter: DualDTrainingAdapter
    classifier: Classifier


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve requested device name into a torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


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


def build_datasets(args):
    """Build source train, target train, and target validation datasets."""

    source_train = MultiModalDomainDataset(
        root_dir=args.source_root,
        domain_type="source",
        phase=args.train_phase,
        layout=args.source_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        image_size=args.image_size,
        resize_size=args.resize_size,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
    )
    label_map = source_train.get_label_map()
    target_train = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.train_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
    )
    target_val = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.val_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        val_augment=args.val_augment,
        augmentation_strength=getattr(args, "augmentation_strength", 0.0),
    )
    target_train_eval = MultiModalDomainDataset(
        root_dir=args.target_root,
        domain_type="target",
        phase=args.train_phase,
        layout=args.target_layout,
        vis_folder=args.vis_folder,
        ir_folder=args.ir_folder,
        global_label_map=label_map,
        image_size=args.image_size,
        resize_size=args.resize_size,
        val_augment=False,
        train_augment=False,
        augmentation_strength=0.0,
    )
    return source_train, target_train, target_train_eval, target_val, label_map


def build_models(args, num_classes: int, device: torch.device) -> ModelBundle:
    """Instantiate all standalone Dual_D model modules."""

    dual_config = load_config(args.dual_config)
    fused_dim = args.proj_dim * 2
    if dual_config.feature_dim != fused_dim:
        dual_config.feature_dim = fused_dim

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
    tal = TensorBasedAlignmentStable(
        input_dims=[args.feature_dim, args.feature_dim],
        output_dims=[args.proj_dim, args.proj_dim],
        num_modalities=2,
    ).to(device)
    dual_adapter = DualDTrainingAdapter(dual_config).to(device)
    classifier = Classifier(
        input_dim=fused_dim,
        num_classes=num_classes,
        dropout=getattr(args, "classifier_dropout", 0.30),
    ).to(device)
    return ModelBundle(net_vis, net_ir, tal, dual_adapter, classifier)


def build_optimizers(args, models: ModelBundle):
    """Build main and discriminator optimizers."""

    visual_params = [parameter for parameter in models.net_vis.parameters() if parameter.requires_grad]
    main_params = [
        *[parameter for parameter in models.net_ir.parameters() if parameter.requires_grad],
        *list(models.tal.parameters()),
        *list(models.dual_adapter.generator_parameters()),
        *list(models.classifier.parameters()),
    ]

    param_groups = []
    if visual_params:
        param_groups.append({"params": visual_params, "lr": args.lr_visual})
    param_groups.append({"params": main_params, "lr": args.lr_main})

    optimizer_main = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    optimizer_disc = optim.AdamW(
        models.dual_adapter.discriminator_parameters(),
        lr=args.lr_discriminator,
        weight_decay=args.weight_decay,
    )
    return optimizer_main, optimizer_disc


def extract_fused_features(
    models: ModelBundle,
    source_batch: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract TAL-aligned fused source and target features."""

    source_vis = source_batch["vis"].to(device)
    source_ir = source_batch["ir"].to(device)
    target_vis = target_batch["vis"].to(device)
    target_ir = target_batch["ir"].to(device)

    source_vis_feat = models.net_vis(source_vis)
    source_ir_feat = models.net_ir(source_ir)
    target_vis_feat = models.net_vis(target_vis)
    target_ir_feat = models.net_ir(target_ir)

    projected_source, projected_target, loss_tal = models.tal(
        [source_vis_feat, source_ir_feat],
        [target_vis_feat, target_ir_feat],
    )
    feat_src = torch.cat(projected_source, dim=1)
    feat_tgt = torch.cat(projected_target, dim=1)
    return feat_src, feat_tgt, loss_tal


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


def train_one_epoch(
    args,
    models: ModelBundle,
    paired_loader: PairedClassSampler,
    optimizer_main,
    optimizer_disc,
    criterion_cls: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train all Dual_D components for one epoch."""

    models.net_vis.train()
    models.net_ir.train()
    models.tal.train()
    models.dual_adapter.train()
    models.classifier.train()
    adversarial_scale = _adversarial_scale(args, epoch)

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
    }

    for step, (source_batch, target_batch) in enumerate(paired_loader, start=1):
        source_labels = source_batch["label"].to(device)
        target_labels = target_batch["label"].to(device)
        labels_for_contrast = source_labels

        feat_src, feat_tgt, loss_tal = extract_fused_features(
            models,
            source_batch,
            target_batch,
            device,
        )
        dual_outputs = models.dual_adapter.forward_features(
            feat_src,
            feat_tgt,
            labels=labels_for_contrast,
        )

        if adversarial_scale > 0 and step % args.discriminator_update_interval == 0:
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
            _accumulate_logs(totals, d_logs)
            totals["disc_steps"] += 1.0

        models.dual_adapter.set_discriminators_trainable(False)
        optimizer_main.zero_grad(set_to_none=True)

        pred_src = models.classifier(feat_src)
        pred_tgt = models.classifier(feat_tgt)
        loss_cls_source = criterion_cls(pred_src, source_labels)
        loss_cls_target = criterion_cls(pred_tgt, target_labels)
        loss_cls = loss_cls_source + loss_cls_target

        loss_dual_g, g_logs = models.dual_adapter.compute_generator_loss(
            outputs=dual_outputs,
            labels=labels_for_contrast,
            classifier=models.classifier,
            criterion_cls=criterion_cls,
            source_labels=source_labels,
            target_labels=target_labels,
            adversarial_scale=adversarial_scale,
        )
        _accumulate_logs(totals, g_logs)
        loss_total = loss_cls + args.tal_weight * loss_tal + loss_dual_g
        if not bool(torch.isfinite(loss_total)):
            raise FloatingPointError(
                f"Non-finite main loss at epoch={epoch}, step={step}."
            )
        loss_total.backward()

        main_parameters = [
            *[p for p in models.net_vis.parameters() if p.requires_grad],
            *[p for p in models.net_ir.parameters() if p.requires_grad],
            *list(models.tal.parameters()),
            *list(models.dual_adapter.generator_parameters()),
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

        optimizer_main.step()
        models.tal.apply_orthogonal_projection()
        models.dual_adapter.set_discriminators_trainable(True)

        with torch.no_grad():
            pred_source_labels = torch.argmax(pred_src.detach(), dim=1)
            pred_target_labels = torch.argmax(pred_tgt.detach(), dim=1)
            source_like_logits = models.classifier(dual_outputs.source_like.detach())
            target_like_logits = models.classifier(dual_outputs.target_like.detach())
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
        "train_adversarial_scale": adversarial_scale,
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
        "train_dual_d_classification_feedback": _average_logged_metric(
            totals,
            "dual_d_classification_feedback",
            steps,
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
) -> Dict[str, object]:
    """Evaluate target-domain validation accuracy and metrics."""

    models.net_vis.eval()
    models.net_ir.eval()
    models.tal.eval()
    models.dual_adapter.eval()
    models.classifier.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        vis = batch["vis"].to(device)
        ir = batch["ir"].to(device)
        labels = batch["label"].to(device)

        vis_feat = models.net_vis(vis)
        ir_feat = models.net_ir(ir)
        projected_target = models.tal.project_target([vis_feat, ir_feat])
        features = torch.cat(projected_target, dim=1)
        selected_mode = feature_mode or args.eval_feature_mode
        if selected_mode != "raw":
            features = models.dual_adapter.inference_features(
                features,
                mode=selected_mode,
            )
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
    metrics["feature_mode"] = feature_mode or args.eval_feature_mode
    return metrics


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
        "tal": models.tal.state_dict(),
        "dual_adapter": models.dual_adapter.state_dict(),
        "classifier": models.classifier.state_dict(),
        "optimizer_main": optimizer_main.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
    }


def run_training(args) -> Dict[str, object]:
    """Run a complete standalone Dual_D training experiment."""

    set_seed(args.seed)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"dual_d_{Path(args.target_root).name}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_text_logger(run_dir / "train.log")
    metrics_logger = CSVMetricLogger(run_dir / "metrics.csv")

    logger.info("Starting Dual_D standalone training")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")

    source_train, target_train, target_train_eval, target_val, label_map = build_datasets(args)
    num_classes = len(label_map)
    save_json({"args": vars(args), "label_map": label_map}, run_dir / "resolved_config.json")
    save_json(label_map, run_dir / "label_map.json")

    logger.info(f"Source train samples: {len(source_train)}")
    logger.info(f"Target train samples: {len(target_train)}")
    logger.info(f"Target val samples: {len(target_val)}")
    logger.info(f"Classes: {num_classes}")

    data_audit = audit_dataset_splits(
        target_train,
        target_val,
        hash_contents=bool(getattr(args, "data_audit_hashes", False)),
    )
    save_json(data_audit, run_dir / "data_audit.json")
    audit_errors = data_audit_errors(data_audit)
    logger.info(
        "Data audit: same_dir=%s | path_overlap(vis/ir)=%d/%d | "
        "content_overlap(vis/ir)=%d/%d | stem_mismatch=%d",
        data_audit["same_base_dir"],
        data_audit["path_overlap_vis_count"],
        data_audit["path_overlap_ir_count"],
        data_audit["content_overlap_vis_count"],
        data_audit["content_overlap_ir_count"],
        data_audit["vis_ir_stem_mismatch_count"],
    )
    if audit_errors:
        message = "Data audit failed: " + "; ".join(audit_errors)
        if bool(getattr(args, "strict_data_audit", True)):
            raise RuntimeError(message)
        logger.warning(message)

    class_summaries = {
        "source_train": summarize_label_distribution(source_train, num_classes, label_map),
        "target_train": summarize_label_distribution(target_train, num_classes, label_map),
        "target_val": summarize_label_distribution(target_val, num_classes, label_map),
    }
    for split_name, split_summary in class_summaries.items():
        logger.info(
            "%s classes: present %d/%d | absent %d | counts [%s]",
            split_name,
            split_summary["present_class_count"],
            split_summary["num_classes"],
            split_summary["absent_class_count"],
            _compact_distribution(split_summary),
        )

    paired_loader = PairedClassSampler(source_train, target_train, args.batch_size)
    paired_common_classes = list(paired_loader.classes)
    save_json(
        {
            "label_map": label_map,
            "paired_training_classes": paired_common_classes,
            **class_summaries,
        },
        run_dir / "class_distribution.json",
    )
    logger.info(
        "Paired training classes: %d/%d | class ids [%s]",
        len(paired_common_classes),
        num_classes,
        ", ".join(str(class_id) for class_id in paired_common_classes),
    )
    val_loader = DataLoader(
        target_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        target_train_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    models = build_models(args, num_classes, device)
    total_parameters = sum(parameter.numel() for model in models.__dict__.values() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for model in models.__dict__.values()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "Model parameters: total=%d | trainable=%d",
        total_parameters,
        trainable_parameters,
    )
    criterion_cls = LabelSmoothingCrossEntropy(eps=args.label_smoothing)
    optimizer_main, optimizer_disc = build_optimizers(args, models)
    monitor_metric = getattr(args, "monitor_metric", "val_f1_macro_present")
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
    best_score = float("inf") if monitor_mode == "min" else float("-inf")
    best_metrics: Dict[str, object] = {}
    start_time = time.time()
    epochs_without_improvement = 0
    epochs_completed = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            args=args,
            models=models,
            paired_loader=paired_loader,
            optimizer_main=optimizer_main,
            optimizer_disc=optimizer_disc,
            criterion_cls=criterion_cls,
            device=device,
            epoch=epoch,
        )
        val_metrics = evaluate(
            args=args,
            models=models,
            dataloader=val_loader,
            criterion_cls=criterion_cls,
            device=device,
            num_classes=num_classes,
        )
        if args.eval_feature_mode == "raw":
            val_raw_metrics = val_metrics
        else:
            val_raw_metrics = evaluate(
                args=args,
                models=models,
                dataloader=val_loader,
                criterion_cls=criterion_cls,
                device=device,
                num_classes=num_classes,
                feature_mode="raw",
            )

        train_eval_interval = max(int(getattr(args, "train_eval_interval", 1)), 1)
        if epoch == 1 or epoch % train_eval_interval == 0:
            train_full_metrics = evaluate(
                args=args,
                models=models,
                dataloader=train_eval_loader,
                criterion_cls=criterion_cls,
                device=device,
                num_classes=num_classes,
            )
        else:
            train_full_metrics = {}

        monitor_values = {
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1_macro_present": float(val_metrics["f1_macro_present"]),
            "val_loss": float(val_metrics["val_loss"]),
        }
        monitor_value = monitor_values[monitor_metric]
        scheduler_main.step(monitor_value)
        scheduler_disc.step(monitor_value)

        if args.eval_feature_mode == "raw":
            sampled_mode_acc = float(train_metrics["train_acc_target"])
        elif args.eval_feature_mode == "source_like":
            sampled_mode_acc = float(train_metrics["train_acc_source_like"])
        else:
            sampled_mode_acc = float("nan")
        full_train_acc = train_full_metrics.get("accuracy")

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
            "val_raw_loss": val_raw_metrics["val_loss"],
            "val_raw_acc": val_raw_metrics["accuracy"],
            "val_raw_f1_macro_present": val_raw_metrics["f1_macro_present"],
            "train_full_acc": full_train_acc,
            "train_full_f1_macro_present": train_full_metrics.get("f1_macro_present"),
            "train_sampled_minus_full_acc": (
                sampled_mode_acc - float(full_train_acc)
                if full_train_acc is not None and np.isfinite(sampled_mode_acc)
                else None
            ),
            "monitor_value": monitor_value,
            "lr_main": optimizer_main.param_groups[-1]["lr"],
            "lr_discriminator": optimizer_disc.param_groups[0]["lr"],
            "lr_discriminator_to_main_ratio": (
                optimizer_disc.param_groups[0]["lr"]
                / max(optimizer_main.param_groups[-1]["lr"], 1e-30)
            ),
            "epoch_seconds": time.time() - epoch_start,
        }
        metrics_logger.write_row(row)

        logger.info(
            "Epoch %03d/%03d | loss %.4f | cls %.4f | tal %.4f | dual_g %.4f | "
            "dual_d %.4f | train_acc %.4f | train_full %s | val_acc %.4f | "
            "val_f1 %.4f | grad(main/disc) %.3f/%.3f | lr(main/disc) %.2e/%.2e | %.1fs",
            epoch,
            args.epochs,
            row["train_loss"],
            row["train_loss_cls"],
            row["train_loss_tal"],
            row["train_loss_dual_g"],
            row["train_loss_dual_d"],
            row["train_acc"],
            f"{row['train_full_acc']:.4f}" if row["train_full_acc"] is not None else "n/a",
            row["val_acc"],
            row["val_f1_macro_present"],
            row["train_grad_norm_main"],
            row["train_grad_norm_discriminator"],
            row["lr_main"],
            row["lr_discriminator"],
            row["epoch_seconds"],
        )

        last_state = checkpoint_state(
            args,
            models,
            optimizer_main,
            optimizer_disc,
            epoch,
            {"train": train_metrics, "val": val_metrics},
            label_map,
        )
        save_checkpoint(last_state, run_dir / "checkpoints" / "last_model.pt")

        best_acc = max(best_acc, float(val_metrics["accuracy"]))
        min_delta = float(getattr(args, "early_stopping_min_delta", 0.0))
        improved = (
            monitor_value < best_score - min_delta
            if monitor_mode == "min"
            else monitor_value > best_score + min_delta
        )
        if improved:
            best_score = monitor_value
            epochs_without_improvement = 0
            best_metrics = {
                "train": train_metrics,
                "train_full": train_full_metrics,
                "val": val_metrics,
                "val_raw": val_raw_metrics,
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "epoch": epoch,
            }
            save_checkpoint(last_state, run_dir / "checkpoints" / "best_model.pt")
            save_json(best_metrics, run_dir / "best_metrics.json")
            logger.info(
                "New best %s: %.4f at epoch %d",
                monitor_metric,
                monitor_value,
                epoch,
            )
        else:
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

    summary = {
        "run_dir": str(run_dir),
        "best_acc": best_acc,
        "best_monitor_metric": monitor_metric,
        "best_monitor_value": best_score,
        "best_metrics": best_metrics,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "total_seconds": time.time() - start_time,
    }
    save_json(summary, run_dir / "result_summary.json")
    logger.info(
        "Training complete. Best validation accuracy: %.4f | best %s: %.4f",
        best_acc,
        monitor_metric,
        best_score,
    )
    close_text_logger(logger)
    return summary
