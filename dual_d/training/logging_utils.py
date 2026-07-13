"""Logging helpers for standalone Dual_D training runs.

Module purpose:
    Configure text logging and append epoch metrics to CSV files. This keeps the
    training entrypoint small and makes experiment outputs easy to parse.

Public interfaces:
    - setup_text_logger(log_path)
    - CSVMetricLogger(csv_path)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable


def setup_text_logger(log_path: str | Path) -> logging.Logger:
    """Create a console/file logger for one training run."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"DualDTraining:{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_text_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers owned by one run logger."""

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class CSVMetricLogger:
    """Append dictionaries of scalar metrics to a CSV file."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = None

    def write_row(self, row: Dict[str, object]) -> None:
        """Append one metric row."""

        scalar_row = {
            key: value
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        if self.fieldnames is None:
            self.fieldnames = list(scalar_row.keys())
            write_header = not self.csv_path.exists()
        else:
            write_header = False

        with self.csv_path.open("a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({field: scalar_row.get(field) for field in self.fieldnames})
