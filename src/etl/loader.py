"""
Data Loader — Async bulk loader with warehouse integration and deduplication
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Async data loader with support for bulk insert, upsert, and deduplication.

    Example:
        >>> loader = DataLoader()
        >>> count = await loader.load(df, "warehouse.data_records")
    """

    def __init__(
        self,
        warehouse_manager=None,
        dry_run: bool = False,
        dedup_on: Optional[list[str]] = None,
    ) -> None:
        self.warehouse = warehouse_manager
        self.dry_run = dry_run
        self.dedup_on = dedup_on or ["_checksum"]
        self._loaded_checksums: set[str] = set()

    async def load(self, df: pd.DataFrame, destination: str) -> int:
        """Load DataFrame into destination table. Returns count of loaded records."""
        if df.empty:
            return 0

        # Deduplication check
        if "_checksum" in df.columns:
            before = len(df)
            df = df[~df["_checksum"].isin(self._loaded_checksums)]
            dupes = before - len(df)
            if dupes > 0:
                logger.debug(f"Deduplication removed {dupes:,} duplicate records")
            self._loaded_checksums.update(df["_checksum"].tolist())

        if df.empty:
            return 0

        if self.dry_run:
            logger.info(f"[DRY RUN] Would load {len(df):,} records → {destination}")
            return len(df)

        if self.warehouse:
            loaded = await self.warehouse.bulk_insert(destination, df)
        else:
            # Fallback: log-only mode for testing
            logger.info(f"Loaded {len(df):,} records → {destination}")
            loaded = len(df)

        return loaded

    async def flush(self) -> None:
        """Flush any pending batches and clear dedup cache."""
        self._loaded_checksums.clear()
        logger.debug("Loader flushed")
