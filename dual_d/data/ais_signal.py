"""AIS signal loading utilities for the three-modality Dual-D pipeline.

The loader accepts one AIS file per VIS/IR sample.  Supported formats are
``.npy``, ``.npz``, ``.csv``, ``.txt`` and ``.json``.  Every sample is
converted to a fixed-size ``[2, sequence_length]`` tensor:

- native complex arrays are split into in-phase (I) and quadrature (Q) parts;
- real arrays with a two-channel axis are interpreted as I/Q data;
- one-dimensional real feature vectors are stored in I and receive a zero Q
  channel, which keeps them compatible with both the complex and MLP encoders.

This representation follows the I/Q AIS description in the JMDA-Net paper
while still allowing pre-extracted numerical AIS attributes on a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


AIS_EXTENSIONS = {".npy", ".npz", ".csv", ".txt", ".json"}


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


__all__ = ["AIS_EXTENSIONS", "ais_files", "load_ais_signal"]
