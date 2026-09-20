"""Export one deterministic paired Optical/SAR example per M4-SAR class.

The two modalities are written as separate PNG files.  They are nevertheless
guaranteed to come from the same manifest record and to use the same crop box,
which preserves the physical correspondence without constructing a montage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Iterable

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.data.m4sar_classification import (  # noqa: E402
    M4SAR_CLASS_NAMES,
    M4SAR_OPTICAL_MEAN,
    M4SAR_OPTICAL_STD,
    M4SAR_SAR_MEAN,
    M4SAR_SAR_STD,
    M4SARClassificationDataset,
)


def _safe_name(value: str) -> str:
    """Return a filesystem-safe, still-auditable pair id."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return cleaned or "pair"


def _denormalize(
    tensor: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    """Convert a normalized CHW model tensor back to displayable uint8."""

    channels = int(tensor.shape[0])
    mean_tensor = torch.tensor(tuple(mean), dtype=tensor.dtype).view(channels, 1, 1)
    std_tensor = torch.tensor(tuple(std), dtype=tensor.dtype).view(channels, 1, 1)
    restored = (tensor.detach().cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    array = restored.mul(255.0).round().to(torch.uint8).numpy()
    return array


def _save_optical(tensor: torch.Tensor, path: Path) -> None:
    array = _denormalize(tensor, M4SAR_OPTICAL_MEAN, M4SAR_OPTICAL_STD)
    Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB").save(path)


def _save_sar(tensor: torch.Tensor, path: Path) -> None:
    array = _denormalize(tensor, M4SAR_SAR_MEAN, M4SAR_SAR_STD)
    Image.fromarray(array[0], mode="L").save(path)


def _select_one_index_per_class(dataset: M4SARClassificationDataset, seed: int) -> dict[int, int]:
    candidates = {class_id: [] for class_id in range(len(M4SAR_CLASS_NAMES))}
    for index, record in enumerate(dataset.records):
        candidates[int(record.label)].append(index)
    missing = [class_id for class_id, indices in candidates.items() if not indices]
    if missing:
        raise ValueError(
            f"split={dataset.split!r} lacks required M4-SAR classes: {missing}."
        )
    rng = random.Random(int(seed))
    return {class_id: rng.choice(indices) for class_id, indices in candidates.items()}


def export_examples(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    split: str = "val",
    domain: str = "target",
    input_size: int = 128,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    """Export separate but paired modality files and return their metadata."""

    domains = ("source", "target") if domain == "both" else (str(domain),)
    if any(item not in {"source", "target"} for item in domains):
        raise ValueError("domain must be one of: source, target, both")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite to replace generated files."
        )
    output.mkdir(parents=True, exist_ok=True)

    base_dataset = M4SARClassificationDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        domain=domains[0],
        input_size=int(input_size),
        augment=False,
    )
    selected = _select_one_index_per_class(base_dataset, seed)
    datasets = {
        item: base_dataset.with_domain(item, augment=False)
        for item in domains
    }

    examples = []
    for class_id, class_name in enumerate(M4SAR_CLASS_NAMES):
        index = selected[class_id]
        record = base_dataset.records[index]
        class_entry = {
            "class_id": class_id,
            "class_name": class_name,
            "pair_id": record.pair_id,
            "crop_box": list(record.crop_box),
            "domains": {},
        }
        stem = f"{class_id:02d}_{class_name}__{_safe_name(record.pair_id)}.png"
        for domain_name, dataset in datasets.items():
            sample = dataset[index]
            if sample["pair_id"] != record.pair_id or int(sample["label"]) != class_id:
                raise AssertionError("Dataset views no longer reference the selected manifest pair.")
            optical_dir = output / domain_name / "optical"
            sar_dir = output / domain_name / "sar"
            optical_dir.mkdir(parents=True, exist_ok=True)
            sar_dir.mkdir(parents=True, exist_ok=True)
            optical_path = optical_dir / stem
            sar_path = sar_dir / stem
            _save_optical(sample["optical"], optical_path)
            _save_sar(sample["sar"], sar_path)
            source_paths = (
                (record.source_optical_path, record.source_sar_path)
                if domain_name == "source"
                else (record.target_optical_path, record.target_sar_path)
            )
            class_entry["domains"][domain_name] = {
                "scene_id": int(sample["scene_id"]),
                "optical_png": str(optical_path.resolve()),
                "sar_png": str(sar_path.resolve()),
                "original_optical_path": source_paths[0],
                "original_sar_path": source_paths[1],
            }
        examples.append(class_entry)

    metadata = {
        "manifest": str(Path(manifest_path).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "output_dir": str(output.resolve()),
        "split": str(split),
        "domains": list(domains),
        "input_size": int(input_size),
        "selection_seed": int(seed),
        "augmentation": False,
        "selection_policy": "one deterministic manifest record per class",
        "pairing_policy": "Optical and SAR share pair_id, record index, and crop_box",
        "display_policy": (
            "Dataset tensors after crop/resize/source-normalization are inverse-normalized "
            "only for lossless human-viewable PNG export"
        ),
        "examples": examples,
    }
    metadata_path = output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return metadata


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="每类导出一组严格对应、但分文件保存的 M4-SAR Optical/SAR 样例。"
    )
    parser.add_argument("--config", default="configs/m4sar_classification_optimized.json")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--domain", choices=("source", "target", "both"), default="target")
    parser.add_argument("--input-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    manifest = args.manifest or str(config.get("m4sar_manifest", ""))
    data_root = args.data_root or str(config.get("m4sar_data_root", ""))
    input_size = args.input_size or int(config.get("m4sar_input_size", 128))
    if not manifest:
        parser.error("M4-SAR manifest is missing; set --manifest or m4sar_manifest in config.")
    if not data_root:
        parser.error("M4-SAR data root is missing; set --data-root or m4sar_data_root in config.")
    output_dir = args.output_dir or str(
        Path("runs")
        / "m4sar_class_examples"
        / f"{args.split}_{args.domain}_seed{args.seed}"
    )
    metadata = export_examples(
        manifest_path=manifest,
        data_root=data_root,
        output_dir=output_dir,
        split=args.split,
        domain=args.domain,
        input_size=input_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": "ok",
        "output_dir": metadata["output_dir"],
        "split": metadata["split"],
        "domains": metadata["domains"],
        "classes": len(metadata["examples"]),
        "png_files": len(metadata["examples"]) * len(metadata["domains"]) * 2,
        "metadata": str((Path(output_dir) / "metadata.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
