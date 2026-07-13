"""Dataset-integrity checks for standalone Dual_D training.

The audit is intentionally independent of model code. It detects accidental
train/validation reuse, exact content duplicates, missing files, and suspicious
VIS/IR filename pairing before an experiment is allowed to report metrics.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _resolved_paths(records, attribute: str) -> set[str]:
    """Return normalized absolute paths for one record attribute."""

    return {str(Path(getattr(record, attribute)).resolve()) for record in records}


def _file_digest(path: Path, cache: Dict[str, str]) -> str:
    """Return a cached SHA-256 digest for one file."""

    key = str(path.resolve())
    if key not in cache:
        digest = sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def _content_overlaps(train_paths: Iterable[Path], val_paths: Iterable[Path]) -> Tuple[int, List[str]]:
    """Count validation files whose exact content also appears in training."""

    cache: Dict[str, str] = {}
    train_hashes = {_file_digest(Path(path), cache) for path in train_paths}
    overlaps = []
    for path in val_paths:
        path = Path(path)
        if _file_digest(path, cache) in train_hashes:
            overlaps.append(str(path))
    return len(overlaps), overlaps[:10]


def audit_dataset_splits(
    train_dataset,
    val_dataset,
    hash_contents: bool = False,
) -> Dict[str, object]:
    """Audit target train/validation splits and return JSON-serializable evidence.

    Path overlap is always checked. Exact SHA-256 content overlap is optional
    because it reads every train/validation image once.
    """

    train_records = list(getattr(train_dataset, "samples", []))
    val_records = list(getattr(val_dataset, "samples", []))
    train_vis = _resolved_paths(train_records, "vis_path")
    train_ir = _resolved_paths(train_records, "ir_path")
    val_vis = _resolved_paths(val_records, "vis_path")
    val_ir = _resolved_paths(val_records, "ir_path")

    path_overlap_vis = sorted(train_vis & val_vis)
    path_overlap_ir = sorted(train_ir & val_ir)
    same_base_dir = Path(train_dataset.base_dir).resolve() == Path(val_dataset.base_dir).resolve()

    all_records = train_records + val_records
    missing_files = [
        str(path)
        for record in all_records
        for path in (record.vis_path, record.ir_path)
        if not Path(path).is_file()
    ]
    label_path_mismatches = [
        {
            "raw_label": record.raw_label,
            "vis_path": str(record.vis_path),
            "ir_path": str(record.ir_path),
        }
        for record in all_records
        if record.raw_label not in record.vis_path.parts
        or record.raw_label not in record.ir_path.parts
    ]
    stem_mismatch_count = sum(
        int(record.vis_path.stem != record.ir_path.stem) for record in all_records
    )

    audit: Dict[str, object] = {
        "train_base_dir": str(Path(train_dataset.base_dir).resolve()),
        "val_base_dir": str(Path(val_dataset.base_dir).resolve()),
        "same_base_dir": same_base_dir,
        "train_sample_count": len(train_records),
        "val_sample_count": len(val_records),
        "path_overlap_vis_count": len(path_overlap_vis),
        "path_overlap_ir_count": len(path_overlap_ir),
        "path_overlap_vis_examples": path_overlap_vis[:10],
        "path_overlap_ir_examples": path_overlap_ir[:10],
        "missing_file_count": len(missing_files),
        "missing_file_examples": missing_files[:10],
        "label_path_mismatch_count": len(label_path_mismatches),
        "label_path_mismatch_examples": label_path_mismatches[:10],
        "vis_ir_stem_mismatch_count": stem_mismatch_count,
        "content_hash_checked": bool(hash_contents),
        "content_overlap_vis_count": 0,
        "content_overlap_ir_count": 0,
        "content_overlap_vis_examples": [],
        "content_overlap_ir_examples": [],
    }

    if hash_contents:
        vis_count, vis_examples = _content_overlaps(
            [record.vis_path for record in train_records],
            [record.vis_path for record in val_records],
        )
        ir_count, ir_examples = _content_overlaps(
            [record.ir_path for record in train_records],
            [record.ir_path for record in val_records],
        )
        audit.update(
            {
                "content_overlap_vis_count": vis_count,
                "content_overlap_ir_count": ir_count,
                "content_overlap_vis_examples": vis_examples,
                "content_overlap_ir_examples": ir_examples,
            }
        )

    audit["leakage_detected"] = bool(
        same_base_dir
        or path_overlap_vis
        or path_overlap_ir
        or audit["content_overlap_vis_count"]
        or audit["content_overlap_ir_count"]
    )
    return audit


def data_audit_errors(audit: Dict[str, object]) -> List[str]:
    """Convert audit findings that invalidate evaluation into clear messages."""

    errors = []
    if audit.get("same_base_dir"):
        errors.append("target train and validation resolve to the same directory")
    if audit.get("path_overlap_vis_count") or audit.get("path_overlap_ir_count"):
        errors.append("target train and validation share image paths")
    if audit.get("content_overlap_vis_count") or audit.get("content_overlap_ir_count"):
        errors.append("target train and validation contain byte-identical images")
    if audit.get("missing_file_count"):
        errors.append("dataset records reference missing image files")
    if audit.get("label_path_mismatch_count"):
        errors.append("a sample label does not match its class directory")
    return errors
