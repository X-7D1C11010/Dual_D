"""Lazy HDF5 dataset support for So2Sat LCZ42 v4.

The dataset keeps the registered Sentinel-1/Sentinel-2 patch pair intact and
uses normalization statistics computed from the official ``training.h5``
source split.  Image arrays are opened lazily in each DataLoader worker; only
labels and selected row indices are retained in memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SO2SAT_LCZ42_CLASS_NAMES: Tuple[str, ...] = (
    "Compact high-rise",
    "Compact mid-rise",
    "Compact low-rise",
    "Open high-rise",
    "Open mid-rise",
    "Open low-rise",
    "Lightweight low-rise",
    "Large low-rise",
    "Sparsely built",
    "Heavy industry",
    "Dense trees",
    "Scattered trees",
    "Bush/scrub",
    "Low plants",
    "Bare rock/paved",
    "Bare soil/sand",
    "Water",
)

S1_MEAN: Tuple[float, ...] = (
    -3.591224256609e-05,
    -7.658561276843e-06,
    5.937385747597e-05,
    2.516623153712e-05,
    4.420110659759e-02,
    2.576102708500e-01,
    7.556743372573e-04,
    1.350346683002e-03,
)
S1_STD: Tuple[float, ...] = (
    0.175552011374,
    0.175564632750,
    0.459987934178,
    0.455988755730,
    2.855990921313,
    8.324800606440,
    2.449875738256,
    1.464735298451,
)
S2_MEAN: Tuple[float, ...] = (
    0.123756961177,
    0.109277463637,
    0.101085520327,
    0.114239861611,
    0.159265669202,
    0.181472360088,
    0.174574031229,
    0.195016073496,
    0.154284688721,
    0.109050506996,
)
S2_STD: Tuple[float, ...] = (
    0.039587959859,
    0.047778262752,
    0.066366167064,
    0.063588749125,
    0.077443871480,
    0.091016350859,
    0.092184665624,
    0.101645812339,
    0.099917730435,
    0.087806325091,
)


def so2sat_label_map() -> Dict[str, int]:
    """Return the fixed official LCZ label mapping."""

    return {name: class_id for class_id, name in enumerate(SO2SAT_LCZ42_CLASS_NAMES)}


def _class_ids(one_hot_labels: np.ndarray, path: Path) -> np.ndarray:
    """Validate the official one-hot matrix and return contiguous class ids."""

    labels = np.asarray(one_hot_labels)
    if labels.ndim != 2 or labels.shape[1] != len(SO2SAT_LCZ42_CLASS_NAMES):
        raise ValueError(
            f"Expected label shape [N, 17] in {path}, got {labels.shape}."
        )
    if not np.isfinite(labels).all():
        raise ValueError(f"Non-finite labels found in {path}.")
    row_sums = labels.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Labels in {path} are not one-hot rows.")
    return labels.argmax(axis=1).astype(np.int64, copy=False)


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic class-stratified train/validation row indices."""

    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    label_array = np.asarray(labels, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in sorted(np.unique(label_array).tolist()):
        class_indices = np.flatnonzero(label_array == class_id)
        generator.shuffle(class_indices)
        validation_count = max(1, int(round(len(class_indices) * fraction)))
        if validation_count >= len(class_indices):
            validation_count = len(class_indices) - 1
        if validation_count <= 0:
            raise ValueError(
                f"Class {class_id} has too few samples for a stratified split."
            )
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())
    train_indices.sort()
    validation_indices.sort()
    return train_indices, validation_indices


class So2SatLCZ42Dataset(Dataset):
    """Index-based registered Sentinel-1/Sentinel-2 LCZ classification dataset."""

    def __init__(
        self,
        data_path: str | Path,
        geo_path: str | Path | None,
        domain_role: str,
        indices: Iterable[int] | None = None,
        augment: bool = False,
        sar_clip_after_normalize: float | None = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.geo_path = Path(geo_path) if geo_path else None
        self.domain_role = str(domain_role)
        if self.domain_role not in {"source", "target_adapt", "target_test"}:
            raise ValueError(f"Unsupported domain_role: {self.domain_role}")
        self.domain_label = 0 if self.domain_role == "source" else 1
        self.augment = bool(augment)
        self.sar_clip_after_normalize = (
            None
            if sar_clip_after_normalize is None
            else float(sar_clip_after_normalize)
        )
        if (
            self.sar_clip_after_normalize is not None
            and self.sar_clip_after_normalize <= 0.0
        ):
            raise ValueError("sar_clip_after_normalize must be positive or None.")
        if not self.data_path.is_file():
            raise FileNotFoundError(f"So2Sat data file does not exist: {self.data_path}")
        if self.geo_path is not None and not self.geo_path.is_file():
            raise FileNotFoundError(f"So2Sat geo file does not exist: {self.geo_path}")

        with h5py.File(self.data_path, "r") as data_file:
            missing = {"sen1", "sen2", "label"} - set(data_file.keys())
            if missing:
                raise KeyError(
                    f"Missing So2Sat datasets in {self.data_path}: {sorted(missing)}"
                )
            sample_count = int(data_file["label"].shape[0])
            if tuple(data_file["sen1"].shape[1:]) != (32, 32, 8):
                raise ValueError(
                    f"Expected sen1 [N,32,32,8], got {data_file['sen1'].shape}."
                )
            if tuple(data_file["sen2"].shape[1:]) != (32, 32, 10):
                raise ValueError(
                    f"Expected sen2 [N,32,32,10], got {data_file['sen2'].shape}."
                )
            if int(data_file["sen1"].shape[0]) != sample_count or int(
                data_file["sen2"].shape[0]
            ) != sample_count:
                raise ValueError("So2Sat sen1/sen2/label row counts do not match.")
            all_labels = _class_ids(data_file["label"][:], self.data_path)

        if indices is None:
            selected = np.arange(sample_count, dtype=np.int64)
        else:
            selected = np.asarray(list(indices), dtype=np.int64)
            if selected.ndim != 1:
                raise ValueError("indices must be a one-dimensional sequence.")
            if selected.size and (selected.min() < 0 or selected.max() >= sample_count):
                raise IndexError("So2Sat row index is outside the HDF5 dataset.")
            if np.unique(selected).size != selected.size:
                raise ValueError("indices must not contain duplicates.")
        self.indices = selected
        self.labels = all_labels[selected].tolist()
        self.label_map = so2sat_label_map()
        self.base_dir = self.data_path

        if self.geo_path is not None:
            with h5py.File(self.geo_path, "r") as geo_file:
                if "city" not in geo_file:
                    raise KeyError(f"Missing city dataset in {self.geo_path}.")
                if int(geo_file["city"].shape[0]) != sample_count:
                    raise ValueError("So2Sat data and geo row counts do not match.")

        self._data_file: h5py.File | None = None
        self._geo_file: h5py.File | None = None
        self._owner_pid: int | None = None
        self._sar_mean = torch.tensor(S1_MEAN, dtype=torch.float32).view(8, 1, 1)
        self._sar_std = torch.tensor(S1_STD, dtype=torch.float32).view(8, 1, 1)
        self._optical_mean = torch.tensor(S2_MEAN, dtype=torch.float32).view(10, 1, 1)
        self._optical_std = torch.tensor(S2_STD, dtype=torch.float32).view(10, 1, 1)

    def get_label_map(self) -> Dict[str, int]:
        """Return the fixed raw-label to class-id map."""

        return dict(self.label_map)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _close_handles(self) -> None:
        for handle_name in ("_data_file", "_geo_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                finally:
                    setattr(self, handle_name, None)
        self._owner_pid = None

    def close(self) -> None:
        """Close HDF5 handles owned by the current process."""

        self._close_handles()

    def _ensure_open(self) -> None:
        process_id = os.getpid()
        if self._owner_pid != process_id:
            self._close_handles()
        if self._data_file is None:
            self._data_file = h5py.File(self.data_path, "r")
        if self.geo_path is not None and self._geo_file is None:
            self._geo_file = h5py.File(self.geo_path, "r")
        self._owner_pid = process_id

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data_file"] = None
        state["_geo_file"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        self._close_handles()

    @staticmethod
    def _decode_city(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
        return str(value)

    def _synchronized_geometry(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return sar, optical
        if bool(torch.rand(()) < 0.5):
            sar = torch.flip(sar, dims=(2,))
            optical = torch.flip(optical, dims=(2,))
        if bool(torch.rand(()) < 0.5):
            sar = torch.flip(sar, dims=(1,))
            optical = torch.flip(optical, dims=(1,))
        rotations = int(torch.randint(0, 4, ()).item())
        if rotations:
            sar = torch.rot90(sar, rotations, dims=(1, 2))
            optical = torch.rot90(optical, rotations, dims=(1, 2))
        return sar.contiguous(), optical.contiguous()

    def __getitem__(self, index: int):
        self._ensure_open()
        row_index = int(self.indices[int(index)])
        assert self._data_file is not None
        sar_array = np.asarray(self._data_file["sen1"][row_index], dtype=np.float32)
        optical_array = np.asarray(self._data_file["sen2"][row_index], dtype=np.float32)
        if not np.isfinite(sar_array).all() or not np.isfinite(optical_array).all():
            raise FloatingPointError(
                f"Non-finite So2Sat input at {self.data_path.name}:{row_index}."
            )

        sar = torch.from_numpy(np.ascontiguousarray(sar_array.transpose(2, 0, 1)))
        optical = torch.from_numpy(
            np.ascontiguousarray(optical_array.transpose(2, 0, 1))
        )
        sar = (sar - self._sar_mean) / self._sar_std
        optical = (optical - self._optical_mean) / self._optical_std
        if self.sar_clip_after_normalize is not None:
            clip = self.sar_clip_after_normalize
            sar = sar.clamp(min=-clip, max=clip)
        sar, optical = self._synchronized_geometry(sar, optical)

        city = ""
        if self._geo_file is not None:
            city = self._decode_city(self._geo_file["city"][row_index])
        label = int(self.labels[int(index)])
        return {
            "sar": sar,
            "optical": optical,
            "label": torch.tensor(label, dtype=torch.long),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "domain_role": self.domain_role,
            "city": city,
            "index": torch.tensor(row_index, dtype=torch.long),
            "sample_id": f"{self.data_path.name}:{row_index}",
        }
