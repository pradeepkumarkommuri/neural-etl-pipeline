"""
Neural ETL Pipeline - Core Orchestrator
High-performance ETL pipeline with TensorFlow-powered anomaly detection
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

import pandas as pd

from src.etl.extractor import DataExtractor
from src.etl.transformer import DataTransformer
from src.etl.loader import DataLoader
from src.utils.metrics import PipelineMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PipelineConfig:
    name: str
    batch_size: int = 10_000
    max_retries: int = 3
    retry_delay: float = 2.0
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 0.95
    parallel_workers: int = 4
    checkpoint_interval: int = 50_000
    enable_metrics: bool = True
    dry_run: bool = False


@dataclass
class PipelineResult:
    pipeline_name: str
    status: PipelineStatus
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_rejected: int = 0
    anomalies_detected: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.records_extracted == 0:
            return 0.0
        return (self.records_loaded / self.records_extracted) * 100

    @property
    def throughput(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.records_loaded / self.duration_seconds


class NeuralETLPipeline:
    """
    Production-grade ETL pipeline with TensorFlow-powered ML anomaly detection.

    Features:
    - Async batch processing with configurable parallelism
    - Real-time anomaly detection using autoencoders
    - Automatic retry with exponential backoff
    - Checkpoint/resume capability
    - Comprehensive metrics and observability
    - SQL-native data warehouse integration

    Example:
        >>> config = PipelineConfig(name="sales_pipeline", batch_size=5000)
        >>> pipeline = NeuralETLPipeline(config)
        >>> result = await pipeline.run(source="sales_db", destination="warehouse")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.status = PipelineStatus.IDLE
        self._metrics = PipelineMetrics(config.name) if config.enable_metrics else None
        self._checkpoint_store: dict[str, Any] = {}
        self._hooks: dict[str, list[Callable]] = {
            "pre_extract": [],
            "post_extract": [],
            "pre_transform": [],
            "post_transform": [],
            "pre_load": [],
            "post_load": [],
        }

        # Initialize components
        self.extractor = DataExtractor(batch_size=config.batch_size)
        self.transformer = DataTransformer()
        self.loader = DataLoader(dry_run=config.dry_run)

        if config.enable_anomaly_detection:
            from src.models.anomaly_detector import AnomalyDetector
            self.anomaly_detector = AnomalyDetector(
                threshold=config.anomaly_threshold
            )
        else:
            self.anomaly_detector = None

        logger.info(f"Pipeline '{config.name}' initialized | batch_size={config.batch_size}")

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register lifecycle hooks for pipeline events."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event '{event}'. Valid: {list(self._hooks.keys())}")
        self._hooks[event].append(callback)
        logger.debug(f"Hook registered for event '{event}'")

    async def _execute_hooks(self, event: str, data: Any) -> Any:
        for hook in self._hooks[event]:
            if asyncio.iscoroutinefunction(hook):
                data = await hook(data)
            else:
                data = hook(data)
        return data

    async def run(
        self,
        source: str,
        destination: str,
        filters: Optional[dict] = None,
        resume_from_checkpoint: bool = False,
    ) -> PipelineResult:
        """Execute the full ETL pipeline with monitoring and error handling."""

        result = PipelineResult(
            pipeline_name=self.config.name,
            status=PipelineStatus.RUNNING,
        )
        start_time = time.perf_counter()
        self.status = PipelineStatus.RUNNING

        if self._metrics:
            self._metrics.pipeline_start()

        logger.info(f"Starting pipeline '{self.config.name}' | source={source} → dest={destination}")

        try:
            checkpoint_offset = 0
            if resume_from_checkpoint:
                checkpoint_offset = self._checkpoint_store.get("last_offset", 0)
                logger.info(f"Resuming from checkpoint offset={checkpoint_offset}")

            async for batch_result in self._process_batches(
                source=source,
                destination=destination,
                filters=filters,
                start_offset=checkpoint_offset,
            ):
                result.records_extracted += batch_result["extracted"]
                result.records_transformed += batch_result["transformed"]
                result.records_loaded += batch_result["loaded"]
                result.records_rejected += batch_result["rejected"]
                result.anomalies_detected += batch_result["anomalies"]

                if self._metrics:
                    self._metrics.record_batch(batch_result)

                # Checkpoint
                if result.records_extracted % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(result.records_extracted)

            result.status = PipelineStatus.COMPLETED
            self.status = PipelineStatus.COMPLETED

        except Exception as exc:
            result.status = PipelineStatus.FAILED
            result.errors.append(str(exc))
            self.status = PipelineStatus.FAILED
            logger.error(f"Pipeline failed: {exc}", exc_info=True)

        finally:
            result.duration_seconds = time.perf_counter() - start_time
            if self._metrics:
                self._metrics.pipeline_end(result)
            self._log_summary(result)

        return result

    async def _process_batches(
        self,
        source: str,
        destination: str,
        filters: Optional[dict],
        start_offset: int,
    ) -> AsyncGenerator[dict, None]:
        """Core async batch processing loop with retry logic."""

        batch_num = 0
        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async for raw_batch in self.extractor.stream(source, filters=filters, offset=start_offset):
            batch_num += 1
            async with semaphore:
                batch_result = await self._process_single_batch(
                    batch=raw_batch,
                    batch_num=batch_num,
                    destination=destination,
                )
                yield batch_result

    async def _process_single_batch(
        self,
        batch: pd.DataFrame,
        batch_num: int,
        destination: str,
    ) -> dict[str, int]:
        """Process a single batch through the E→T→L stages with retry."""

        for attempt in range(self.config.max_retries):
            try:
                # EXTRACT hook
                batch = await self._execute_hooks("post_extract", batch)

                # TRANSFORM
                await self._execute_hooks("pre_transform", batch)
                transformed, rejected = await self.transformer.transform(batch)
                transformed = await self._execute_hooks("post_transform", transformed)

                # ANOMALY DETECTION
                anomaly_count = 0
                if self.anomaly_detector and len(transformed) > 0:
                    clean, anomalies = await self.anomaly_detector.filter(transformed)
                    anomaly_count = len(anomalies)
                    transformed = clean
                    if anomaly_count > 0:
                        logger.warning(f"Batch {batch_num}: {anomaly_count} anomalies quarantined")

                # LOAD
                await self._execute_hooks("pre_load", transformed)
                loaded_count = await self.loader.load(transformed, destination)
                await self._execute_hooks("post_load", transformed)

                return {
                    "extracted": len(batch),
                    "transformed": len(transformed),
                    "loaded": loaded_count,
                    "rejected": len(rejected),
                    "anomalies": anomaly_count,
                }

            except Exception as exc:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Batch {batch_num} attempt {attempt+1} failed: {exc}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Batch {batch_num} permanently failed after {self.config.max_retries} attempts")
                    raise

        return {"extracted": 0, "transformed": 0, "loaded": 0, "rejected": 0, "anomalies": 0}

    def _save_checkpoint(self, offset: int) -> None:
        self._checkpoint_store["last_offset"] = offset
        self._checkpoint_store["timestamp"] = datetime.utcnow().isoformat()
        logger.debug(f"Checkpoint saved at offset={offset}")

    def _log_summary(self, result: PipelineResult) -> None:
        logger.info(
            f"Pipeline '{result.pipeline_name}' {result.status.value.upper()} | "
            f"extracted={result.records_extracted:,} | "
            f"loaded={result.records_loaded:,} | "
            f"rejected={result.records_rejected:,} | "
            f"anomalies={result.anomalies_detected:,} | "
            f"success_rate={result.success_rate:.1f}% | "
            f"throughput={result.throughput:.0f} rec/s | "
            f"duration={result.duration_seconds:.2f}s"
        )
