"""On-the-fly target-crop classification dataset for M4-SAR.

The classification manifest contains one row per valid Source/Target object
pair.  Pair metadata is retained for auditing and error analysis only; callers
select exactly one domain and the training pipeline shuffles Source and Target
with independent DataLoaders.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


M4SAR_CLASS_NAMES = (
    "bridge",
    "harbor",
    "oil_tank",
    "playground",
    "airport",
    "wind_turbine",
)

M4SAR_OPTICAL_MEAN = (
    0.32412607310812486,
    0.3259470182913754,
    0.3061976798789559,
)
M4SAR_OPTICAL_STD = (
    0.19963838987820892,
    0.1975738079640792,
    0.1994368095218188,
)
M4SAR_SAR_MEAN = (0.47364379746860025,)
M4SAR_SAR_STD = (0.24904880500119383,)
M4SAR_TRAIN_CLASS_COUNTS = (105400, 6768, 88030, 38388, 783, 3554)


def m4sar_label_map() -> Dict[str, int]:
    """Return the fixed six-class label map."""

    return {name: index for index, name in enumerate(M4SAR_CLASS_NAMES)}


def inverse_sqrt_class_weights(
    counts: Sequence[int],
) -> tuple[float, ...]:
    """Return inverse-square-root class weights normalized to mean one."""

    values = np.asarray(tuple(int(value) for value in counts), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("Class counts must be a non-empty sequence of positive integers.")
    weights = 1.0 / np.sqrt(values)
    weights /= weights.mean()
    return tuple(float(value) for value in weights)


M4SAR_CLASS_WEIGHTS = inverse_sqrt_class_weights(M4SAR_TRAIN_CLASS_COUNTS)


@dataclass(frozen=True)
class M4SARRecord:
    """One manifest object pair with a shared crop box."""

    split: str
    label: int
    pair_id: str
    source_scene_id: int
    target_scene_id: int
    source_optical_path: str
    source_sar_path: str
    target_optical_path: str
    target_sar_path: str
    crop_box: tuple[int, int, int, int]


def _first(row: Mapping[str, str], aliases: Iterable[str], *, required: bool = True) -> str:
    for name in aliases:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    if required:
        raise KeyError(f"Manifest is missing one of the required columns: {tuple(aliases)}")
    return ""


def _parse_int(value: str, field: str) -> int:
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {field} value: {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be a finite integer, got {value!r}.")
    return int(numeric)


def _parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in M4SAR_CLASS_NAMES:
        return M4SAR_CLASS_NAMES.index(normalized)
    return _parse_int(value, "class_id")


def _parse_crop_box(row: Mapping[str, str]) -> tuple[int, int, int, int]:
    coordinate_aliases = (
        ("crop_left", "crop_top", "crop_right", "crop_bottom"),
        ("crop_x1", "crop_y1", "crop_x2", "crop_y2"),
        ("crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    )
    for aliases in coordinate_aliases:
        if all(alias in row and str(row[alias]).strip() != "" for alias in aliases):
            values = tuple(_parse_int(str(row[alias]), alias) for alias in aliases)
            break
    else:
        packed = _first(row, ("crop_box", "crop", "bbox", "crop_xyxy"))
        try:
            decoded = json.loads(packed)
            if isinstance(decoded, dict):
                decoded = [
                    decoded[key]
                    for key in ("left", "top", "right", "bottom")
                ]
        except (json.JSONDecodeError, KeyError):
            decoded = [part for part in re.split(r"[\s,;|]+", packed.strip("[]()")) if part]
        if not isinstance(decoded, (list, tuple)) or len(decoded) != 4:
            raise ValueError(f"Crop box must contain four coordinates, got {packed!r}.")
        values = tuple(_parse_int(str(value), "crop_box") for value in decoded)

    left, top, right, bottom = values
    if left < 0 or top < 0 or right > 512 or bottom > 512:
        raise ValueError(f"Manifest crop box is outside the 512x512 patch: {values}.")
    if right <= left or bottom <= top:
        raise ValueError(f"Manifest crop box has non-positive size: {values}.")
    return left, top, right, bottom


def _resolve_manifest_path(value: str, data_root: Path, manifest_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    rooted = data_root / path
    # In production the data root and manifest directory differ. Check that
    # cheap condition first to avoid millions of filesystem stat calls while
    # parsing the 485k-row manifest.
    if data_root != manifest_dir or rooted.exists():
        return str(rooted)
    return str(manifest_dir / path)


def load_m4sar_manifest(
    manifest_path: str | Path,
    split: str,
    data_root: str | Path | None = None,
) -> tuple[M4SARRecord, ...]:
    """Read and validate one official split from a CSV or CSV.GZ manifest."""

    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"M4-SAR manifest does not exist: {manifest}")
    root = Path(data_root) if data_root else manifest.parent
    requested_split = str(split).strip().lower()
    if requested_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported M4-SAR split: {split}")

    opener = gzip.open if manifest.suffix.lower() == ".gz" else open
    records = []
    with opener(manifest, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"M4-SAR manifest has no header: {manifest}")
        for row_number, row in enumerate(reader, start=2):
            row_split = _first(row, ("split", "official_split", "subset")).lower()
            if row_split in {"validation", "valid"}:
                row_split = "val"
            if row_split != requested_split:
                continue
            try:
                label = _parse_label(
                    _first(row, ("class_id", "label", "class", "category_id")),
                )
                if label < 0 or label >= len(M4SAR_CLASS_NAMES):
                    raise ValueError(f"class_id must be in [0,5], got {label}.")
                source_scene_id = _parse_int(
                    _first(
                        row,
                        ("source_scene_id", "src_scene_id", "source_id", "source_scene"),
                    ),
                    "source_scene_id",
                )
                target_scene_id = _parse_int(
                    _first(
                        row,
                        ("target_scene_id", "tgt_scene_id", "target_id", "target_scene"),
                    ),
                    "target_scene_id",
                )
                if not 1 <= source_scene_id <= 56087:
                    raise ValueError(
                        f"source_scene_id must be in [1,56087], got {source_scene_id}."
                    )
                if target_scene_id != source_scene_id + 56087:
                    raise ValueError(
                        "M4-SAR pair boundary mismatch: target_scene_id must equal "
                        f"source_scene_id + 56087, got {source_scene_id}/{target_scene_id}."
                    )
                pair_id = _first(
                    row,
                    ("pair_id", "object_pair_id", "sample_id", "object_id"),
                    required=False,
                ) or f"{source_scene_id}:{row_number}"
                records.append(
                    M4SARRecord(
                        split=requested_split,
                        label=label,
                        pair_id=pair_id,
                        source_scene_id=source_scene_id,
                        target_scene_id=target_scene_id,
                        source_optical_path=_resolve_manifest_path(
                            _first(row, ("source_optical_path", "src_optical_path", "source_optical")),
                            root,
                            manifest.parent,
                        ),
                        source_sar_path=_resolve_manifest_path(
                            _first(row, ("source_sar_path", "src_sar_path", "source_sar")),
                            root,
                            manifest.parent,
                        ),
                        target_optical_path=_resolve_manifest_path(
                            _first(row, ("target_optical_path", "tgt_optical_path", "target_optical")),
                            root,
                            manifest.parent,
                        ),
                        target_sar_path=_resolve_manifest_path(
                            _first(row, ("target_sar_path", "tgt_sar_path", "target_sar")),
                            root,
                            manifest.parent,
                        ),
                        crop_box=_parse_crop_box(row),
                    )
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid M4-SAR manifest row {row_number} in {manifest}: {error}"
                ) from error
    if not records:
        raise ValueError(f"M4-SAR manifest contains no rows for split={requested_split!r}.")
    return tuple(records)


class M4SARClassificationDataset(Dataset):
    """Select one M4-SAR domain and crop Optical/SAR jointly on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        domain: str,
        data_root: str | Path | None = None,
        input_size: int = 128,
        augment: bool = False,
        optical_jitter: float = 0.0,
        sar_gain_jitter: float = 0.0,
        sar_noise_std: float = 0.0,
        records: Sequence[M4SARRecord] | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        self.split = str(split).lower()
        self.domain = str(domain).lower()
        if self.domain not in {"source", "target"}:
            raise ValueError("M4-SAR domain must be 'source' or 'target'.")
        self.domain_role = self.domain
        self.domain_label = 0 if self.domain == "source" else 1
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        self.augment = bool(augment)
        self.optical_jitter = float(optical_jitter)
        self.sar_gain_jitter = float(sar_gain_jitter)
        self.sar_noise_std = float(sar_noise_std)
        for name, value in (
            ("optical_jitter", self.optical_jitter),
            ("sar_gain_jitter", self.sar_gain_jitter),
            ("sar_noise_std", self.sar_noise_std),
        ):
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"{name} must be in [0, 0.5], got {value}.")
        self.records = tuple(records) if records is not None else load_m4sar_manifest(
            self.manifest_path,
            self.split,
            self.data_root,
        )
        if not self.records or any(record.split != self.split for record in self.records):
            raise ValueError("M4-SAR records do not match the requested split.")
        self.labels = [record.label for record in self.records]
        self.label_map = m4sar_label_map()
        self._optical_mean = torch.tensor(M4SAR_OPTICAL_MEAN).view(3, 1, 1)
        self._optical_std = torch.tensor(M4SAR_OPTICAL_STD).view(3, 1, 1)
        self._sar_mean = torch.tensor(M4SAR_SAR_MEAN).view(1, 1, 1)
        self._sar_std = torch.tensor(M4SAR_SAR_STD).view(1, 1, 1)

    def with_domain(self, domain: str, *, augment: bool | None = None):
        """Return a lightweight view that shares immutable manifest records."""

        return M4SARClassificationDataset(
            manifest_path=self.manifest_path,
            split=self.split,
            domain=domain,
            data_root=self.data_root,
            input_size=self.input_size,
            augment=self.augment if augment is None else augment,
            optical_jitter=self.optical_jitter,
            sar_gain_jitter=self.sar_gain_jitter,
            sar_noise_std=self.sar_noise_std,
            records=self.records,
        )

    def get_label_map(self) -> Dict[str, int]:
        return dict(self.label_map)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def _synchronized_geometry(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return optical, sar
        if bool(torch.rand(()) < 0.5):
            optical, sar = torch.flip(optical, (2,)), torch.flip(sar, (2,))
        if bool(torch.rand(()) < 0.5):
            optical, sar = torch.flip(optical, (1,)), torch.flip(sar, (1,))
        rotations = int(torch.randint(0, 4, ()).item())
        if rotations:
            optical = torch.rot90(optical, rotations, (1, 2))
            sar = torch.rot90(sar, rotations, (1, 2))
        return optical.contiguous(), sar.contiguous()

    def _photometric_augmentation(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mild modality-specific radiometric perturbations.

        Geometry remains synchronized within one domain sample.  Optical and
        SAR radiometry is intentionally perturbed independently because their
        sensing physics differs.  Values are clipped before Source-derived
        normalization so the augmentation cannot create impossible byte-range
        intensities.
        """

        if not self.augment:
            return optical, sar
        if self.optical_jitter > 0.0:
            strength = self.optical_jitter
            brightness = 1.0 + float(torch.empty(()).uniform_(-strength, strength))
            contrast = 1.0 + float(torch.empty(()).uniform_(-strength, strength))
            channel_gain = torch.empty((3, 1, 1)).uniform_(
                1.0 - 0.5 * strength,
                1.0 + 0.5 * strength,
            )
            channel_mean = optical.mean(dim=(1, 2), keepdim=True)
            optical = (optical - channel_mean) * contrast + channel_mean
            optical = optical * brightness * channel_gain
            optical = optical.clamp_(0.0, 1.0)
        if self.sar_gain_jitter > 0.0:
            gain = 1.0 + float(
                torch.empty(()).uniform_(
                    -self.sar_gain_jitter,
                    self.sar_gain_jitter,
                )
            )
            sar = sar * gain
        if self.sar_noise_std > 0.0:
            sar = sar + torch.randn_like(sar) * self.sar_noise_std
        return optical.contiguous(), sar.clamp_(0.0, 1.0).contiguous()

    def __getitem__(self, index: int):
        record = self.records[int(index)]
        if self.domain == "source":
            optical_path = record.source_optical_path
            sar_path = record.source_sar_path
            scene_id = record.source_scene_id
        else:
            optical_path = record.target_optical_path
            sar_path = record.target_sar_path
            scene_id = record.target_scene_id

        with Image.open(optical_path) as image:
            optical_image = image.convert("RGB").crop(record.crop_box).resize(
                (self.input_size, self.input_size),
                Image.Resampling.BILINEAR,
            )
            optical = self._to_tensor(optical_image)
        with Image.open(sar_path) as image:
            sar_image = image.convert("L").crop(record.crop_box).resize(
                (self.input_size, self.input_size),
                Image.Resampling.BILINEAR,
            )
            sar = self._to_tensor(sar_image)

        optical, sar = self._synchronized_geometry(optical, sar)
        optical, sar = self._photometric_augmentation(optical, sar)
        optical = (optical - self._optical_mean) / self._optical_std
        sar = (sar - self._sar_mean) / self._sar_std
        label = int(record.label)
        return {
            "optical": optical,
            "sar": sar,
            "label": torch.tensor(label, dtype=torch.long),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "domain": self.domain,
            "domain_role": self.domain,
            "pair_id": record.pair_id,
            "scene_id": torch.tensor(scene_id, dtype=torch.long),
            "sample_id": f"{self.domain}:{record.pair_id}",
            "crop_box": torch.tensor(record.crop_box, dtype=torch.long),
        }
