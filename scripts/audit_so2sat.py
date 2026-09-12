#!/usr/bin/env python3
"""Audit a So2Sat LCZ42 v4 installation before Satellite-Dual_D training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.data import SO2SAT_LCZ42_CLASS_NAMES  # noqa: E402


SPLITS = (
    ("source", "training.h5", "training_geo.h5"),
    ("target_adapt", "validation.h5", "validation_geo.h5"),
    ("target_test", "testing.h5", "testing_geo.h5"),
)


def _decode_cities(values: np.ndarray) -> list[str]:
    return sorted(
        {
            bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        }
    )


def _channel_summary(array: np.ndarray) -> dict[str, list[float] | int]:
    finite = np.isfinite(array)
    axes = tuple(range(array.ndim - 1))
    return {
        "nonfinite_count": int(array.size - finite.sum()),
        "min": np.nanmin(array, axis=axes).astype(float).tolist(),
        "max": np.nanmax(array, axis=axes).astype(float).tolist(),
        "mean": np.nanmean(array, axis=axes).astype(float).tolist(),
        "std": np.nanstd(array, axis=axes).astype(float).tolist(),
    }


def audit_split(
    root: Path,
    role: str,
    data_name: str,
    geo_name: str,
    sample_count: int,
    seed: int,
) -> dict[str, object]:
    data_path = root / data_name
    geo_path = root / geo_name
    if not data_path.is_file() or not geo_path.is_file():
        raise FileNotFoundError(f"Missing {role} files: {data_path}, {geo_path}")

    with h5py.File(data_path, "r") as data_file:
        keys = sorted(data_file.keys())
        if keys != ["label", "sen1", "sen2"]:
            raise ValueError(f"Unexpected keys in {data_path}: {keys}")
        labels = np.asarray(data_file["label"][:])
        class_ids = labels.argmax(axis=1)
        rows = int(labels.shape[0])
        generator = np.random.default_rng(int(seed))
        selected = np.sort(
            generator.choice(rows, size=min(int(sample_count), rows), replace=False)
        )
        sar = np.asarray(data_file["sen1"][selected])
        optical = np.asarray(data_file["sen2"][selected])
        result: dict[str, object] = {
            "role": role,
            "data_path": str(data_path.resolve()),
            "keys": keys,
            "sample_count": rows,
            "label_shape": list(labels.shape),
            "sar_shape": list(data_file["sen1"].shape),
            "optical_shape": list(data_file["sen2"].shape),
            "label_dtype": str(labels.dtype),
            "sar_dtype": str(data_file["sen1"].dtype),
            "optical_dtype": str(data_file["sen2"].dtype),
            "one_hot_valid": bool(
                labels.ndim == 2
                and labels.shape[1] == 17
                and np.isfinite(labels).all()
                and np.allclose(labels.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
            ),
            "classes_present": sorted(np.unique(class_ids).astype(int).tolist()),
            "class_histogram": np.bincount(class_ids, minlength=17).astype(int).tolist(),
            "sampled_input_count": int(selected.size),
            "sampled_sar_channels": _channel_summary(sar),
            "sampled_optical_channels": _channel_summary(optical),
        }

    with h5py.File(geo_path, "r") as geo_file:
        if "city" not in geo_file:
            raise KeyError(f"Missing city key in {geo_path}")
        cities = _decode_cities(np.asarray(geo_file["city"][:]))
        result["geo_path"] = str(geo_path.resolve())
        result["city_count"] = len(cities)
        result["cities"] = cities
        result["geo_rows_match"] = int(geo_file["city"].shape[0]) == result["sample_count"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="So2Sat LCZ42 v4 root directory")
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")

    root = Path(args.root)
    reports = [
        audit_split(root, role, data_name, geo_name, args.sample_count, args.seed)
        for role, data_name, geo_name in SPLITS
    ]
    class_sets = [report["classes_present"] for report in reports]
    report = {
        "dataset": "So2Sat LCZ42 v4",
        "root": str(root.resolve()),
        "class_names": list(SO2SAT_LCZ42_CLASS_NAMES),
        "closed_set_17_classes": all(
            classes == list(range(17)) for classes in class_sets
        ),
        "splits": reports,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

