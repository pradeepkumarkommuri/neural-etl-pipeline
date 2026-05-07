"""
Data Extractor — Multi-source async streaming extractor
Supports PostgreSQL, MySQL, REST APIs, CSV/Parquet files, and Kafka.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlparse

import aiohttp
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SourceConnector:
    """Base class for all data source connectors."""

    async def stream_batches(
        self,
        source: str,
        batch_size: int,
        filters: Optional[dict] = None,
        offset: int = 0,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        raise NotImplementedError


class SQLConnector(SourceConnector):
    """Stream data from PostgreSQL or MySQL using server-side cursors."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

    async def stream_batches(
        self,
        source: str,
        batch_size: int,
        filters: Optional[dict] = None,
        offset: int = 0,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream table data in batches using server-side cursor."""
        await self._ensure_pool()

        where_clause = ""
        params = []
        if filters:
            conditions = [f'"{k}" = ${i+1}' for i, k in enumerate(filters)]
            where_clause = "WHERE " + " AND ".join(conditions)
            params = list(filters.values())

        sql = f"""
        SELECT * FROM {source}
        {where_clause}
        ORDER BY 1
        LIMIT {batch_size} OFFSET {offset}
        """

        current_offset = offset
        async with self._pool.acquire() as conn:
            while True:
                rows = await conn.fetch(
                    sql.replace(f"OFFSET {offset}", f"OFFSET {current_offset}"),
                    *params,
                )
                if not rows:
                    break
                df = pd.DataFrame([dict(r) for r in rows])
                yield df
                current_offset += len(rows)
                if len(rows) < batch_size:
                    break


class FileConnector(SourceConnector):
    """Stream data from CSV, Parquet, or JSON files."""

    SUPPORTED = {".csv", ".parquet", ".json", ".jsonl"}

    async def stream_batches(
        self,
        source: str,
        batch_size: int,
        filters: Optional[dict] = None,
        offset: int = 0,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")

        loop = asyncio.get_event_loop()

        if suffix == ".parquet":
            async for batch in self._stream_parquet(path, batch_size, offset, loop):
                yield batch
        elif suffix in {".csv"}:
            async for batch in self._stream_csv(path, batch_size, offset, loop):
                yield batch
        elif suffix in {".json", ".jsonl"}:
            async for batch in self._stream_json(path, batch_size, offset, loop):
                yield batch

    async def _stream_parquet(self, path, batch_size, offset, loop):
        import pyarrow.parquet as pq
        pf = await loop.run_in_executor(None, pq.ParquetFile, str(path))
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            if offset > 0:
                offset -= len(df)
                if offset >= 0:
                    continue
                df = df.iloc[max(0, offset):]
                offset = 0
            yield df

    async def _stream_csv(self, path, batch_size, offset, loop):
        def read_chunks():
            return pd.read_csv(str(path), chunksize=batch_size, skiprows=range(1, offset + 1) if offset else None)
        chunks = await loop.run_in_executor(None, read_chunks)
        for chunk in chunks:
            yield chunk

    async def _stream_json(self, path, batch_size, offset, loop):
        def load():
            with open(path) as f:
                if path.suffix == ".jsonl":
                    return [json.loads(l) for l in f if l.strip()]
                return json.load(f)
        records = await loop.run_in_executor(None, load)
        records = records[offset:]
        for i in range(0, len(records), batch_size):
            yield pd.DataFrame(records[i:i + batch_size])


class APIConnector(SourceConnector):
    """Stream paginated data from REST APIs."""

    def __init__(self, headers: Optional[dict] = None) -> None:
        self.headers = headers or {}

    async def stream_batches(
        self,
        source: str,
        batch_size: int,
        filters: Optional[dict] = None,
        offset: int = 0,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Paginate through a REST API endpoint."""
        page = (offset // batch_size) + 1

        async with aiohttp.ClientSession(headers=self.headers) as session:
            while True:
                params = {"page": page, "per_page": batch_size, **(filters or {})}
                async with session.get(source, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                records = data if isinstance(data, list) else data.get("data", data.get("results", []))
                if not records:
                    break

                df = pd.DataFrame(records)
                yield df

                if len(records) < batch_size:
                    break
                page += 1

                # Respect rate limits
                await asyncio.sleep(0.1)


class DataExtractor:
    """
    Unified data extractor with auto-detection of source type.

    Supports:
    - PostgreSQL / MySQL (dsn:// prefix)
    - CSV, Parquet, JSON files
    - REST APIs (http:// or https://)

    Example:
        >>> extractor = DataExtractor(batch_size=10_000)
        >>> async for batch in extractor.stream("postgresql://...", filters={"active": True}):
        ...     process(batch)
    """

    def __init__(self, batch_size: int = 10_000) -> None:
        self.batch_size = batch_size
        self._connectors: dict[str, SourceConnector] = {}

    def register_connector(self, scheme: str, connector: SourceConnector) -> None:
        self._connectors[scheme] = connector

    def _resolve_connector(self, source: str) -> SourceConnector:
        parsed = urlparse(source)
        scheme = parsed.scheme.lower()

        if scheme in ("postgresql", "postgres", "mysql"):
            return self._connectors.get("sql", SQLConnector(source))
        elif scheme in ("http", "https"):
            return self._connectors.get("api", APIConnector())
        else:
            return self._connectors.get("file", FileConnector())

    async def stream(
        self,
        source: str,
        filters: Optional[dict] = None,
        offset: int = 0,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        connector = self._resolve_connector(source)
        batch_count = 0
        total_records = 0

        logger.info(f"Extracting from: {source} | batch_size={self.batch_size}")

        async for batch in connector.stream_batches(
            source=source,
            batch_size=self.batch_size,
            filters=filters,
            offset=offset,
        ):
            if batch.empty:
                continue
            batch_count += 1
            total_records += len(batch)
            logger.debug(f"Extracted batch {batch_count}: {len(batch):,} records")
            yield batch

        logger.info(f"Extraction complete | batches={batch_count} | total={total_records:,}")
