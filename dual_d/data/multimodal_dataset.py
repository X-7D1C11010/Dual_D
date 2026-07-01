"""Standalone visible/infrared multimodal dataset loader.

Module purpose:
    Load paired visible-light and infrared images without importing any script
    from another project folder. The loader supports both common directory
    layouts used by JMDA-style domain adaptation experiments.

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
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SampleRecord:
    """One paired multimodal image sample."""

    vis_path: Path
    ir_path: Path
    raw_label: str


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


def build_transforms(
    phase: str,
    image_size: int = 224,
    resize_size: int = 256,
    val_augment: bool = False,
):
    """Build visible and infrared transforms.

    Args:
        phase: Dataset phase, usually ``train`` or ``val``.
        image_size: Final crop/resize size.
        resize_size: Resize side used before random/center crop.
        val_augment: If true, validation uses train-style random augmentation.

    Returns:
        Dictionary with ``vis`` and ``ir`` transform pipelines.
    """

    train_like = phase == "train" or val_augment
    if train_like:
        image_ops = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        image_ops = [
            transforms.Resize((image_size, image_size)),
        ]

    return {
        "vis": transforms.Compose(
            image_ops
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "ir": transforms.Compose(
            image_ops
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    }


class MultiModalDomainDataset(Dataset):
    """Dataset for paired visible/infrared domain samples.

    Args:
        root_dir: Root directory for one domain.
        domain_type: ``source`` or ``target``. Used only for the returned
            domain label.
        phase: Phase name, usually ``train`` or ``val``.
        layout: ``auto``, ``modality_first``, or ``class_first``.
        vis_folder: Folder name for visible-light images.
        ir_folder: Folder name for infrared images.
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
        global_label_map: Optional[Dict[str, int]] = None,
        image_size: int = 224,
        resize_size: int = 256,
        val_augment: bool = False,
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

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No paired VIS/IR samples found under {self.base_dir} "
                f"with layout={self.layout}, vis_folder={vis_folder}, ir_folder={ir_folder}."
            )

        raw_labels = sorted({sample.raw_label for sample in self.samples})
        if global_label_map is None:
            self.label_map = {raw_label: idx for idx, raw_label in enumerate(raw_labels)}
        else:
            self.label_map = dict(global_label_map)

        self.samples = [
            sample for sample in self.samples if sample.raw_label in self.label_map
        ]
        if not self.samples:
            raise RuntimeError(
                "Samples were found, but none matched the provided global_label_map."
            )

        self.labels = [self.label_map[sample.raw_label] for sample in self.samples]
        self.transform = build_transforms(phase, image_size, resize_size, val_augment)

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
            for vis_path, ir_path in zip(vis_files, ir_files):
                records.append(SampleRecord(vis_path, ir_path, raw_label))
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
            for vis_path, ir_path in zip(vis_files, ir_files):
                records.append(SampleRecord(vis_path, ir_path, raw_label))
        return records

    def get_label_map(self) -> Dict[str, int]:
        """Return raw-label to integer-label mapping."""

        return dict(self.label_map)

    def __len__(self) -> int:
        """Return number of paired samples."""

        return len(self.samples)

    def __getitem__(self, index: int):
        """Load and return one paired multimodal sample."""

        sample = self.samples[index]
        try:
            vis_img = Image.open(sample.vis_path).convert("RGB")
            ir_img = Image.open(sample.ir_path).convert("L").convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read paired sample: {sample.vis_path}, {sample.ir_path}"
            ) from exc

        label = self.label_map[sample.raw_label]
        return {
            "vis": self.transform["vis"](vis_img),
            "ir": self.transform["ir"](ir_img),
            "label": torch.tensor(label, dtype=torch.long),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "raw_label": sample.raw_label,
            "vis_path": str(sample.vis_path),
            "ir_path": str(sample.ir_path),
        }

