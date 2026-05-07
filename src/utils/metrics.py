"""Structured logging and pipeline metrics."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class PipelineMetrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self._start: float = 0
        self._batches: list[dict] = []

    def pipeline_start(self) -> None:
        self._start = time.perf_counter()

    def record_batch(self, batch: dict) -> None:
        self._batches.append(batch)

    def pipeline_end(self, result: Any) -> None:
        total = time.perf_counter() - self._start
        logger = get_logger("metrics")
        logger.info(
            f"[METRICS] pipeline={self.name} | "
            f"batches={len(self._batches)} | "
            f"total_time={total:.2f}s"
        )
