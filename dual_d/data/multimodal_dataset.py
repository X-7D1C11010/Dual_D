"""Standalone visible/infrared/AIS multimodal dataset loader.

Module purpose:
    Load paired visible-light and infrared images plus optional AIS signals
    without importing any script from another project folder. The loader
    supports both common directory layouts used by JMDA-style domain
    adaptation experiments.

Supported layouts:
    1. modality_first:
        root/phase/可见光/class_id/*.jpg
        root/phase/红外/class_id/*.jpg

    2. class_first:
        root/phase/class_id/可见光/*.jpg
        root/phase/class_id/红外/*.jpg

Public interface:
    - MultiModalDomainDataset

Usage:
    >>> ds = MultiModalDomainDataset("/data/sunny", phase="train", layout="auto")
    >>> sample = ds[0]
    >>> sample["vis"].shape, sample["ir"].shape, sample["label"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from .ais_signal import (
    REFERENCE_AIS_FILENAME,
    ais_files,
    group_reference_ais_by_label,
    load_ais_signal,
    resolve_reference_ais_file,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SampleRecord:
    """One aligned visible/infrared/AIS sample."""

    vis_path: Path
    ir_path: Path
    raw_label: str
    ais_path: Optional[Path] = None


def _image_files(directory: Path) -> List[Path]:
    """Return sorted image files in a directory."""

    if not directory.exists():
        return []
    return sorted(
        [
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def _phase_root(root_dir: Path, phase: str) -> Path:
    """Use root/phase when it exists, otherwise use root directly."""

    candidate = root_dir / phase
    return candidate if candidate.exists() else root_dir


class PairedImageTransform:
    """Apply shared geometry and modality-specific normalization to a VIS/IR pair.

    Random crop and horizontal-flip parameters are sampled once and reused for
    both modalities. This preserves pixel-level correspondence between the two
    sensors; applying two independent ``Compose`` objects can silently misalign
    an otherwise paired sample.
    """

    def __init__(
        self,
        train_like: bool,
        image_size: int,
        resize_size: int,
        augmentation_strength: float = 0.0,
    ):
        self.train_like = bool(train_like)
        self.image_size = int(image_size)
        self.resize_size = int(resize_size)
        self.augmentation_strength = max(0.0, min(float(augmentation_strength), 1.0))
        strength = self.augmentation_strength
        self.vis_jitter = transforms.ColorJitter(
            brightness=0.25 * strength,
            contrast=0.25 * strength,
            saturation=0.15 * strength,
            hue=0.05 * strength,
        )
        self.ir_jitter = transforms.ColorJitter(
            brightness=0.15 * strength,
            contrast=0.15 * strength,
        )

    def __call__(self, vis_img: Image.Image, ir_img: Image.Image):
        """Transform and return one synchronized visible/infrared pair."""

        if self.train_like:
            output_size = [self.resize_size, self.resize_size]
            vis_img = transform_functional.resize(vis_img, output_size)
            ir_img = transform_functional.resize(ir_img, output_size)
            top, left, height, width = transforms.RandomCrop.get_params(
                vis_img,
                output_size=(self.image_size, self.image_size),
            )
            vis_img = transform_functional.crop(vis_img, top, left, height, width)
            ir_img = transform_functional.crop(ir_img, top, left, height, width)
            if bool(torch.rand(()) < 0.5):
                vis_img = transform_functional.hflip(vis_img)
                ir_img = transform_functional.hflip(ir_img)
            if self.augmentation_strength > 0:
                vis_img = self.vis_jitter(vis_img)
                ir_img = self.ir_jitter(ir_img)
        else:
            output_size = [self.image_size, self.image_size]
            vis_img = transform_functional.resize(vis_img, output_size)
            ir_img = transform_functional.resize(ir_img, output_size)

        vis_tensor = transform_functional.to_tensor(vis_img)
        ir_tensor = transform_functional.to_tensor(ir_img)
        vis_tensor = transform_functional.normalize(
            vis_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        ir_tensor = transform_functional.normalize(
            ir_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        return vis_tensor, ir_tensor


def build_transforms(
    phase: str,
    image_size: int = 224,
    resize_size: int = 256,
    val_augment: bool = False,
    train_augment: bool = True,
    augmentation_strength: float = 0.0,
):
    """Build visible and infrared transforms.

    Args:
        phase: Dataset phase, usually ``train`` or ``val``.
        image_size: Final crop/resize size.
        resize_size: Resize side used before random/center crop.
        val_augment: If true, validation uses train-style random augmentation.

    Returns:
        A paired transform that applies identical geometry to both modalities.
    """

    train_like = (phase == "train" and train_augment) or val_augment
    return PairedImageTransform(
        train_like,
        image_size,
        resize_size,
        augmentation_strength=augmentation_strength,
    )


class MultiModalDomainDataset(Dataset):
    """Dataset for aligned visible/infrared/AIS domain samples.

    Args:
        root_dir: Root directory for one domain.
        domain_type: ``source`` or ``target``. Used only for the returned
            domain label.
        phase: Phase name, usually ``train`` or ``val``.
        layout: ``auto``, ``modality_first``, or ``class_first``.
        vis_folder: Folder name for visible-light images.
        ir_folder: Folder name for infrared images.
        ais_folder: Folder name for AIS data.
        ais_root: Optional separate AIS root. If omitted, AIS is discovered
            near ``root_dir``. The JMDA-Net global MAT file is preferred when
            present; per-sample numeric files remain supported as fallback.
        ais_data_path: Optional explicit JMDA-Net MAT/HDF5 file path.
        ais_match: Match AIS files to VIS/IR by stem, sorted index, or try stem
            first and fall back to sorted index when set to ``auto``.
        ais_sequence_length: Fixed length returned for each two-channel signal.
        ais_normalize: Apply per-sample, per-channel standardization.
        require_ais: Whether each VIS/IR pair also includes AIS data.
        global_label_map: Optional mapping from raw class names to contiguous
            label ids. Pass the source-domain map into the target domain to keep
            labels aligned.
        image_size: Final network input size.
        resize_size: Pre-crop resize size during training.
        val_augment: Whether to augment validation samples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        domain_type: str = "source",
        phase: str = "train",
        layout: str = "auto",
        vis_folder: str = "可见光",
        ir_folder: str = "红外",
        ais_folder: str = "AIS",
        ais_root: Optional[str | Path] = None,
        ais_data_path: Optional[str | Path] = None,
        ais_match: str = "auto",
        ais_sequence_length: int = 128,
        ais_normalize: bool = True,
        require_ais: bool = True,
        global_label_map: Optional[Dict[str, int]] = None,
        image_size: int = 224,
        resize_size: int = 256,
        val_augment: bool = False,
        train_augment: bool = True,
        augmentation_strength: float = 0.0,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.phase = phase
        self.base_dir = _phase_root(self.root_dir, phase)
        self.domain_type = domain_type
        self.domain_label = 0 if domain_type == "source" else 1
        self.layout = self._resolve_layout(layout, vis_folder, ir_folder)
        self.vis_folder = vis_folder
        self.ir_folder = ir_folder
        self.ais_folder = ais_folder
        self.ais_root = Path(ais_root) if ais_root else self.root_dir
        self.ais_data_path = Path(ais_data_path) if ais_data_path else None
        self.ais_base_dir = _phase_root(self.ais_root, phase)
        self.ais_match = str(ais_match).lower()
        if self.ais_match not in {"auto", "stem", "index"}:
            raise ValueError(f"Unsupported AIS match mode: {ais_match}")
        self.ais_sequence_length = int(ais_sequence_length)
        if self.ais_sequence_length <= 0:
            raise ValueError("ais_sequence_length must be positive.")
        self.ais_normalize = bool(ais_normalize)
        self.require_ais = bool(require_ais)
        self.reference_ais_file = None
        self.reference_ais_by_label = {}
        self.ais_signal_length = self.ais_sequence_length
        if self.require_ais:
            reference_root = self.ais_data_path or self.ais_root
            self.reference_ais_file = resolve_reference_ais_file(
                reference_root,
                ais_folder=self.ais_folder,
                filename=REFERENCE_AIS_FILENAME,
            )
            if self.reference_ais_file is not None:
                self.reference_ais_by_label, self.ais_signal_length = (
                    group_reference_ais_by_label(str(self.reference_ais_file.resolve()))
                )

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No aligned {'VIS/IR/AIS' if self.require_ais else 'VIS/IR'} "
                f"samples found under {self.base_dir} "
                f"with layout={self.layout}, vis_folder={vis_folder}, "
                f"ir_folder={ir_folder}, ais_folder={ais_folder}, "
                f"ais_root={self.ais_root}."
            )

        raw_labels = sorted({sample.raw_label for sample in self.samples})
        if global_label_map is None:
            self.label_map = {raw_label: idx for idx, raw_label in enumerate(raw_labels)}
        else:
            self.label_map = dict(global_label_map)

        unknown_labels = sorted(set(raw_labels) - set(self.label_map))
        if unknown_labels:
            raise ValueError(
                "Dataset contains labels absent from the source label map: "
                f"{unknown_labels}. Fix the directory labels instead of silently dropping samples."
            )

        if self.reference_ais_file is not None:
            missing_ais_labels = []
            for raw_label in raw_labels:
                try:
                    ais_label = int(raw_label)
                except ValueError:
                    missing_ais_labels.append(raw_label)
                    continue
                if ais_label not in self.reference_ais_by_label:
                    missing_ais_labels.append(raw_label)
            if missing_ais_labels:
                raise RuntimeError(
                    "JMDA-Net AIS MAT file has no samples for image labels: "
                    f"{missing_ais_labels}. The image/AIS class numbering must match."
                )

        self.labels = [self.label_map[sample.raw_label] for sample in self.samples]
        self.transform = build_transforms(
            phase,
            image_size,
            resize_size,
            val_augment,
            train_augment,
            augmentation_strength,
        )

    def _resolve_layout(self, layout: str, vis_folder: str, ir_folder: str) -> str:
        """Resolve automatic layout detection."""

        if layout != "auto":
            if layout not in {"modality_first", "class_first"}:
                raise ValueError(f"Unsupported layout: {layout}")
            return layout
        if (self.base_dir / vis_folder).exists() and (self.base_dir / ir_folder).exists():
            return "modality_first"
        return "class_first"

    def _collect_samples(self) -> List[SampleRecord]:
        """Collect paired image records according to the resolved layout."""

        if self.layout == "modality_first":
            return self._collect_modality_first()
        return self._collect_class_first()

    def _collect_modality_first(self) -> List[SampleRecord]:
        """Collect samples from root/phase/modality/class layout."""

        records: List[SampleRecord] = []
        vis_root = self.base_dir / self.vis_folder
        ir_root = self.base_dir / self.ir_folder
        class_dirs = sorted([item for item in vis_root.iterdir() if item.is_dir()])
        for vis_class_dir in class_dirs:
            raw_label = vis_class_dir.name
            ir_class_dir = ir_root / raw_label
            if not ir_class_dir.is_dir():
                continue
            vis_files = _image_files(vis_class_dir)
            ir_files = _image_files(ir_class_dir)
            records.extend(self._align_class_samples(raw_label, vis_files, ir_files))
        return records

    def _collect_class_first(self) -> List[SampleRecord]:
        """Collect samples from root/phase/class/modality layout."""

        records: List[SampleRecord] = []
        class_dirs = sorted([item for item in self.base_dir.iterdir() if item.is_dir()])
        for class_dir in class_dirs:
            raw_label = class_dir.name
            vis_dir = class_dir / self.vis_folder
            ir_dir = class_dir / self.ir_folder
            if not vis_dir.is_dir() or not ir_dir.is_dir():
                continue
            vis_files = _image_files(vis_dir)
            ir_files = _image_files(ir_dir)
            records.extend(self._align_class_samples(raw_label, vis_files, ir_files))
        return records

    def _ais_class_directory(self, raw_label: str) -> Path:
        """Resolve a class AIS directory across common server layouts."""

        candidates = [
            self.ais_base_dir / self.ais_folder / raw_label,
            self.ais_base_dir / raw_label / self.ais_folder,
            self.ais_base_dir / raw_label,
        ]
        for candidate in candidates:
            if ais_files(candidate):
                return candidate
        return candidates[0]

    @staticmethod
    def _pair_images(
        vis_files: Sequence[Path],
        ir_files: Sequence[Path],
    ) -> List[Tuple[Path, Path]]:
        """Preserve the established sorted-index VIS/IR pairing convention."""

        return list(zip(vis_files, ir_files))

    def _align_class_samples(
        self,
        raw_label: str,
        vis_files: Sequence[Path],
        ir_files: Sequence[Path],
    ) -> List[SampleRecord]:
        """Align one class of VIS/IR pairs, optionally with real AIS files."""

        image_pairs = self._pair_images(vis_files, ir_files)
        if not image_pairs:
            return []

        if not self.require_ais or self.reference_ais_file is not None:
            return [
                SampleRecord(vis_path, ir_path, raw_label)
                for vis_path, ir_path in image_pairs
            ]

        ais_dir = self._ais_class_directory(raw_label)
        class_ais_files = ais_files(ais_dir)
        if not class_ais_files:
            raise RuntimeError(
                f"No AIS files found for class={raw_label!r}. Expected files under "
                f"{ais_dir} or pass a separate AIS root."
            )

        by_stem = {path.stem: path for path in class_ais_files}
        stem_matches = [
            by_stem.get(vis_path.stem) or by_stem.get(ir_path.stem)
            for vis_path, ir_path in image_pairs
        ]
        can_use_stems = all(match is not None for match in stem_matches)
        if self.ais_match == "stem" and not can_use_stems:
            missing = [
                vis_path.stem
                for (vis_path, _), match in zip(image_pairs, stem_matches)
                if match is None
            ]
            raise RuntimeError(
                f"AIS stem matching failed for class={raw_label!r}; missing stems: "
                f"{missing[:10]}"
            )

        use_stems = self.ais_match == "stem" or (
            self.ais_match == "auto" and can_use_stems
        )
        if use_stems:
            matched_ais = [match for match in stem_matches if match is not None]
        else:
            if len(class_ais_files) < len(image_pairs):
                raise RuntimeError(
                    f"AIS count is smaller than VIS/IR count for class={raw_label!r}: "
                    f"ais={len(class_ais_files)}, image_pairs={len(image_pairs)}."
                )
            matched_ais = class_ais_files[: len(image_pairs)]

        return [
            SampleRecord(vis_path, ir_path, raw_label, ais_path)
            for (vis_path, ir_path), ais_path in zip(image_pairs, matched_ais)
        ]

    def get_label_map(self) -> Dict[str, int]:
        """Return raw-label to integer-label mapping."""

        return dict(self.label_map)

    def __len__(self) -> int:
        """Return number of paired samples."""

        return len(self.samples)

    def __getitem__(self, index: int):
        """Load one aligned sample in explicit two- or three-modality mode."""

        sample = self.samples[index]
        try:
            vis_img = Image.open(sample.vis_path).convert("RGB")
            ir_img = Image.open(sample.ir_path).convert("L").convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read paired sample: {sample.vis_path}, {sample.ir_path}"
            ) from exc

        label = self.label_map[sample.raw_label]
        vis_tensor, ir_tensor = self.transform(vis_img, ir_img)
        result = {
            "vis": vis_tensor,
            "ir": ir_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "raw_label": sample.raw_label,
            "vis_path": str(sample.vis_path),
            "ir_path": str(sample.ir_path),
            "ais_path": (
                str(sample.ais_path)
                if sample.ais_path is not None
                else str(self.reference_ais_file or "")
            ),
        }
        if self.require_ais:
            try:
                if self.reference_ais_file is not None:
                    ais_label = int(sample.raw_label)
                    class_pool = self.reference_ais_by_label[ais_label]
                    ais_array = class_pool[index % len(class_pool)]
                    result["ais"] = torch.from_numpy(
                        np.ascontiguousarray(ais_array, dtype=np.float32)
                    )
                else:
                    if sample.ais_path is None:
                        raise RuntimeError(
                            f"AIS path is missing for sample: {sample.vis_path}"
                        )
                    result["ais"] = load_ais_signal(
                        sample.ais_path,
                        sequence_length=self.ais_sequence_length,
                        normalize=self.ais_normalize,
                    )
            except Exception as exc:
                source = self.reference_ais_file or sample.ais_path
                raise RuntimeError(f"Failed to read AIS sample: {source}") from exc
        return result
