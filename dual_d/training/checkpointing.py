"""Checkpoint and JSON persistence helpers.

Module purpose:
    Save model checkpoints, resolved training configuration, label maps, and
    final summaries for standalone Dual_D runs.

Public interfaces:
    - save_json(data, path)
    - save_checkpoint(state, path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as UTF-8 JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a torch checkpoint and create parent directories as needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)

