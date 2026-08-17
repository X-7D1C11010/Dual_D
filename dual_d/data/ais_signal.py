"""AIS signal loading utilities for the three-modality Dual-D pipeline.

The preferred input is the JMDA-Net global MATLAB v7.3 file
``balanced_AIS-dataset_16classes_100persample.mat``.  It stores
``balanced_rcv_I``, ``balanced_rcv_Q`` and ``new_balanced_label``; the loader
groups those real I/Q waveforms by class and returns ``[N, 2, L]`` arrays.
One AIS file per VIS/IR sample remains supported as a fallback.  Supported
fallback formats are ``.npy``, ``.npz``, ``.csv``, ``.txt`` and ``.json``.
Every fallback sample is converted to a fixed-size ``[2, sequence_length]`` tensor:

- native complex arrays are split into in-phase (I) and quadrature (Q) parts;
- real arrays with a two-channel axis are interpreted as I/Q data;
- one-dimensional real feature vectors are stored in I and receive a zero Q
  channel, which keeps them compatible with both the complex and MLP encoders.

This representation follows the I/Q AIS description in the JMDA-Net paper
while still allowing pre-extracted numerical AIS attributes on a server.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


AIS_EXTENSIONS = {".npy", ".npz", ".csv", ".txt", ".json"}
REFERENCE_AIS_FILENAME = "balanced_AIS-dataset_16classes_100persample.mat"


def ais_files(directory: Path) -> list[Path]:
    """Return sorted supported AIS files directly under ``directory``."""

    if not directory.is_dir():
        return []
    return sorted(
        [
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in AIS_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def resolve_reference_ais_file(
    root: str | Path,
    ais_folder: str = "AIS",
    filename: str = REFERENCE_AIS_FILENAME,
) -> Path | None:
    """Find the JMDA-Net global AIS MAT file near a domain or AIS root."""

    root = Path(root)
    if root.is_file():
        if root.suffix.lower() not in {".mat", ".h5", ".hdf5"}:
            return None
        return root
    if not root.exists():
        return None

    candidates = []
    current = root.resolve()
    for _ in range(4):
        candidates.extend(
            [
                current / filename,
                current / ais_folder / filename,
            ]
        )
        if current.parent == current:
            break
        current = current.parent

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".mat", ".h5", ".hdf5"}:
            return candidate
    return None


def _mat_array(container, key: str) -> np.ndarray:
    """Read one array from scipy or h5py containers."""

    value = container[key]
    return np.asarray(value[:] if hasattr(value, "shape") and hasattr(value, "__getitem__") else value)


def _find_mat_key(keys, fragment: str) -> str | None:
    """Find a MATLAB/HDF5 key by case-insensitive name fragment."""

    fragment = fragment.lower()
    return next((key for key in keys if fragment in str(key).lower()), None)


def load_reference_ais_mat(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the JMDA-Net global AIS MAT file as ``[N, 2, L]`` and labels.

    The reference project stores I/Q arrays either as ``[L, N]`` or ``[N, L]``.
    This loader preserves the per-sample I/Q waveform and follows the reference
    key convention: ``balanced_rcv_I``, ``balanced_rcv_Q`` and
    ``new_balanced_label``.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"AIS MAT file does not exist: {path}")

    i_data = q_data = labels = None
    errors = []
    try:
        import h5py

        with h5py.File(path, "r") as container:
            keys = list(container.keys())
            i_key = _find_mat_key(keys, "rcv_i")
            q_key = _find_mat_key(keys, "rcv_q")
            label_key = _find_mat_key(keys, "label")
            if i_key and q_key and label_key:
                i_data = _mat_array(container, i_key)
                q_data = _mat_array(container, q_key)
                labels = _mat_array(container, label_key)
    except Exception as exc:
        errors.append(f"hdf5: {exc}")

    if i_data is None or q_data is None or labels is None:
        try:
            import scipy.io as sio

            container = sio.loadmat(path)
            keys = [key for key in container if not str(key).startswith("__")]
            i_key = _find_mat_key(keys, "rcv_i")
            q_key = _find_mat_key(keys, "rcv_q")
            label_key = _find_mat_key(keys, "label")
            if i_key and q_key and label_key:
                i_data = np.asarray(container[i_key])
                q_data = np.asarray(container[q_key])
                labels = np.asarray(container[label_key])
        except Exception as exc:
            errors.append(f"scipy: {exc}")

    if i_data is None or q_data is None or labels is None:
        raise ValueError(
            "Unable to read JMDA-Net AIS keys balanced_rcv_I/balanced_rcv_Q/"
            f"new_balanced_label from {path}. {' | '.join(errors)}"
        )

    i_data = np.asarray(i_data, dtype=np.float32)
    q_data = np.asarray(q_data, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if i_data.ndim != 2 or q_data.ndim != 2 or i_data.shape != q_data.shape:
        raise ValueError(
            f"AIS I/Q arrays must be same-rank 2D arrays, got {i_data.shape} and {q_data.shape}."
        )

    n_samples = labels.size
    if i_data.shape[1] == n_samples:
        i_data = i_data.T
        q_data = q_data.T
    elif i_data.shape[0] != n_samples:
        raise ValueError(
            f"AIS sample count mismatch: I/Q shape={i_data.shape}, labels={labels.shape}."
        )

    if not np.isfinite(labels).all():
        raise ValueError(f"AIS labels contain NaN or Inf: {path}")
    rounded_labels = np.rint(labels)
    if not np.allclose(labels, rounded_labels):
        raise ValueError(f"AIS labels must be integer-valued: {path}")
    labels = rounded_labels.astype(np.int64)
    features = np.stack([i_data, q_data], axis=1)
    if not np.isfinite(features).all():
        raise ValueError(f"AIS data contains NaN or Inf: {path}")
    return np.ascontiguousarray(features, dtype=np.float32), labels


@lru_cache(maxsize=8)
def group_reference_ais_by_label(
    path: str | Path,
) -> Tuple[Dict[int, np.ndarray], int]:
    """Load the global MAT file and group waveform tensors by integer label."""

    features, labels = load_reference_ais_mat(path)
    grouped: Dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        grouped[int(label)] = np.ascontiguousarray(features[labels == label])
    return grouped, int(features.shape[2])


def _first_numeric_json_value(data: Any) -> Any:
    """Resolve common JSON AIS containers to one numerical array-like value."""

    if not isinstance(data, dict):
        return data
    if "i" in data and "q" in data:
        return np.stack([data["i"], data["q"]], axis=0)
    for key in ("iq", "signal", "ais", "features", "values"):
        if key in data:
            return data[key]
    raise ValueError(
        "AIS JSON object must contain i/q, iq, signal, ais, features, or values."
    )


def _load_numeric_array(path: Path) -> np.ndarray:
    """Load one AIS file without permitting pickle-backed object arrays."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"AIS archive is empty: {path}")
            key = next(
                (name for name in ("iq", "signal", "ais", "features", "values") if name in archive),
                archive.files[0],
            )
            return np.asarray(archive[key])
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file_obj:
            return np.asarray(_first_numeric_json_value(json.load(file_obj)))
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32)
        if array.ndim == 2:
            array = array[~np.all(np.isnan(array), axis=1)]
            if array.size:
                array = array[:, ~np.all(np.isnan(array), axis=0)]
        elif array.ndim == 1:
            array = array[~np.isnan(array)]
        return np.asarray(array)
    raise ValueError(f"Unsupported AIS file extension: {path.suffix}")


def _to_iq_channels(array: np.ndarray, path: Path) -> np.ndarray:
    """Convert a complex, I/Q, or real feature array to shape ``[2, length]``."""

    array = np.asarray(array)
    if array.size == 0:
        raise ValueError(f"AIS file contains no numerical values: {path}")

    if np.iscomplexobj(array):
        flat = array.reshape(-1)
        channels = np.stack([flat.real, flat.imag], axis=0)
    else:
        array = np.asarray(array, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 2 and array.shape[0] == 2:
            channels = array.reshape(2, -1)
        elif array.ndim == 2 and array.shape[1] == 2:
            channels = array.T.reshape(2, -1)
        else:
            flat = array.reshape(-1)
            channels = np.stack([flat, np.zeros_like(flat)], axis=0)

    channels = np.asarray(channels, dtype=np.float32)
    channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
    if channels.shape[1] == 0:
        raise ValueError(f"AIS file contains no usable signal samples: {path}")
    return channels


def _resize_signal(channels: np.ndarray, sequence_length: int) -> np.ndarray:
    """Linearly resample I/Q channels to a common sequence length."""

    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("AIS sequence length must be positive.")
    current_length = int(channels.shape[1])
    if current_length == sequence_length:
        return channels
    if current_length == 1:
        return np.repeat(channels, sequence_length, axis=1)
    source_axis = np.linspace(0.0, 1.0, current_length, dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    return np.stack(
        [np.interp(target_axis, source_axis, channel) for channel in channels],
        axis=0,
    ).astype(np.float32)


def load_ais_signal(
    path: str | Path,
    sequence_length: int = 128,
    normalize: bool = True,
) -> torch.Tensor:
    """Load one AIS file as a finite float tensor with shape ``[2, L]``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"AIS file does not exist: {path}")
    channels = _to_iq_channels(_load_numeric_array(path), path)
    channels = _resize_signal(channels, sequence_length)
    if normalize:
        mean = channels.mean(axis=1, keepdims=True)
        std = channels.std(axis=1, keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        channels = (channels - mean) / std
    return torch.from_numpy(np.ascontiguousarray(channels, dtype=np.float32))


__all__ = [
    "AIS_EXTENSIONS",
    "REFERENCE_AIS_FILENAME",
    "ais_files",
    "group_reference_ais_by_label",
    "load_ais_signal",
    "load_reference_ais_mat",
    "resolve_reference_ais_file",
]
