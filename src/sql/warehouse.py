"""
SQL Data Warehouse Manager
Handles schema management, CRUD operations, and query optimization
for the Neural ETL Pipeline data warehouse.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Optional

import asyncpg
import pandas as pd
from asyncpg import Pool

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataType(Enum):
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    FLOAT = "FLOAT8"
    TEXT = "TEXT"
    VARCHAR = "VARCHAR(255)"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMPTZ"
    JSONB = "JSONB"
    UUID = "UUID"


@dataclass
class ColumnDef:
    name: str
    dtype: DataType
    nullable: bool = True
    default: Optional[str] = None
    primary_key: bool = False
    index: bool = False

    def to_sql(self) -> str:
        parts = [f'"{self.name}" {self.dtype.value}']
        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        return " ".join(parts)


@dataclass
class TableSchema:
    name: str
    schema: str
    columns: list[ColumnDef]
    partition_by: Optional[str] = None
    comment: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f'"{self.schema}"."{self.name}"'

    def create_ddl(self) -> str:
        col_defs = ",\n    ".join(col.to_sql() for col in self.columns)
        ddl = f"""
CREATE TABLE IF NOT EXISTS {self.full_name} (
    {col_defs}
){' PARTITION BY RANGE (' + self.partition_by + ')' if self.partition_by else ''};
""".strip()
        if self.comment:
            ddl += f"\nCOMMENT ON TABLE {self.full_name} IS '{self.comment}';"
        return ddl

    def index_ddl(self) -> list[str]:
        ddls = []
        for col in self.columns:
            if col.index and not col.primary_key:
                idx_name = f"idx_{self.name}_{col.name}"
                ddls.append(
                    f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{idx_name}" '
                    f'ON {self.full_name} ("{col.name}");'
                )
        return ddls


# ── Warehouse Schema Definitions ──────────────────────────────────────────

WAREHOUSE_SCHEMAS: list[TableSchema] = [
    TableSchema(
        name="pipeline_runs",
        schema="etl",
        comment="Tracks every pipeline execution",
        columns=[
            ColumnDef("run_id", DataType.UUID, primary_key=True, default="gen_random_uuid()"),
            ColumnDef("pipeline_name", DataType.VARCHAR, nullable=False, index=True),
            ColumnDef("status", DataType.VARCHAR, nullable=False),
            ColumnDef("started_at", DataType.TIMESTAMP, default="NOW()", index=True),
            ColumnDef("completed_at", DataType.TIMESTAMP),
            ColumnDef("records_extracted", DataType.BIGINT, default="0"),
            ColumnDef("records_loaded", DataType.BIGINT, default="0"),
            ColumnDef("records_rejected", DataType.BIGINT, default="0"),
            ColumnDef("anomalies_detected", DataType.BIGINT, default="0"),
            ColumnDef("duration_seconds", DataType.FLOAT),
            ColumnDef("throughput_rps", DataType.FLOAT),
            ColumnDef("error_message", DataType.TEXT),
            ColumnDef("metadata", DataType.JSONB, default="'{}'::jsonb"),
        ],
    ),
    TableSchema(
        name="data_records",
        schema="warehouse",
        comment="Processed and validated data records",
        partition_by="ingested_at",
        columns=[
            ColumnDef("record_id", DataType.UUID, primary_key=True, default="gen_random_uuid()"),
            ColumnDef("source_id", DataType.TEXT, nullable=False, index=True),
            ColumnDef("pipeline_name", DataType.VARCHAR, nullable=False, index=True),
            ColumnDef("ingested_at", DataType.TIMESTAMP, nullable=False, default="NOW()", index=True),
            ColumnDef("data", DataType.JSONB, nullable=False),
            ColumnDef("checksum", DataType.VARCHAR),
        ],
    ),
    TableSchema(
        name="anomaly_log",
        schema="etl",
        comment="Quarantined records flagged by the anomaly detector",
        columns=[
            ColumnDef("id", DataType.BIGINT, primary_key=True),
            ColumnDef("run_id", DataType.UUID, index=True),
            ColumnDef("source_id", DataType.TEXT),
            ColumnDef("detected_at", DataType.TIMESTAMP, default="NOW()", index=True),
            ColumnDef("anomaly_score", DataType.FLOAT, nullable=False),
            ColumnDef("raw_data", DataType.JSONB),
            ColumnDef("resolution_status", DataType.VARCHAR, default="'pending'"),
        ],
    ),
    TableSchema(
        name="batch_metrics",
        schema="etl",
        comment="Per-batch performance metrics",
        columns=[
            ColumnDef("id", DataType.BIGINT, primary_key=True),
            ColumnDef("run_id", DataType.UUID, nullable=False, index=True),
            ColumnDef("batch_number", DataType.INTEGER, nullable=False),
            ColumnDef("batch_size", DataType.INTEGER),
            ColumnDef("extracted", DataType.INTEGER, default="0"),
            ColumnDef("transformed", DataType.INTEGER, default="0"),
            ColumnDef("loaded", DataType.INTEGER, default="0"),
            ColumnDef("rejected", DataType.INTEGER, default="0"),
            ColumnDef("anomalies", DataType.INTEGER, default="0"),
            ColumnDef("duration_ms", DataType.FLOAT),
            ColumnDef("recorded_at", DataType.TIMESTAMP, default="NOW()"),
        ],
    ),
]


# ── Analytical Views (SQL) ────────────────────────────────────────────────

ANALYTICS_VIEWS = {
    "pipeline_summary": """
CREATE OR REPLACE VIEW etl.pipeline_summary AS
SELECT
    pipeline_name,
    COUNT(*)                                        AS total_runs,
    COUNT(*) FILTER (WHERE status = 'completed')   AS successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed')      AS failed_runs,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'completed')::numeric
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                               AS success_rate_pct,
    SUM(records_loaded)                             AS total_records_loaded,
    SUM(anomalies_detected)                         AS total_anomalies,
    ROUND(AVG(throughput_rps)::numeric, 2)          AS avg_throughput_rps,
    MAX(started_at)                                 AS last_run_at
FROM etl.pipeline_runs
GROUP BY pipeline_name
ORDER BY total_records_loaded DESC;
""",
    "hourly_throughput": """
CREATE OR REPLACE VIEW etl.hourly_throughput AS
SELECT
    date_trunc('hour', started_at)  AS hour,
    pipeline_name,
    SUM(records_loaded)             AS records_loaded,
    SUM(anomalies_detected)         AS anomalies,
    ROUND(AVG(throughput_rps)::numeric, 2) AS avg_rps
FROM etl.pipeline_runs
WHERE status = 'completed'
GROUP BY 1, 2
ORDER BY 1 DESC;
""",
    "anomaly_rate_trend": """
CREATE OR REPLACE VIEW etl.anomaly_rate_trend AS
SELECT
    date_trunc('day', started_at)   AS day,
    pipeline_name,
    SUM(anomalies_detected)         AS anomalies,
    SUM(records_extracted)          AS total_extracted,
    ROUND(
        SUM(anomalies_detected)::numeric
        / NULLIF(SUM(records_extracted), 0) * 100, 4
    )                               AS anomaly_rate_pct
FROM etl.pipeline_runs
WHERE status = 'completed'
GROUP BY 1, 2
ORDER BY 1 DESC;
""",
}


class WarehouseManager:
    """
    Async PostgreSQL warehouse manager with connection pooling.

    Handles schema bootstrapping, bulk inserts, and analytical queries.

    Example:
        >>> wm = WarehouseManager("postgresql://user:pass@localhost/db")
        >>> await wm.connect()
        >>> await wm.bootstrap_schema()
        >>> await wm.bulk_insert("warehouse.data_records", df)
    """

    def __init__(self, dsn: str, min_conn: int = 2, max_conn: int = 20) -> None:
        self.dsn = dsn
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        """Establish connection pool."""
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_conn,
            max_size=self.max_conn,
            command_timeout=60,
        )
        logger.info(f"Connection pool created | min={self.min_conn} max={self.max_conn}")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def bootstrap_schema(self) -> None:
        """Create schemas, tables, indexes, and views."""
        async with self._pool.acquire() as conn:
            # Create schemas
            for schema in {"etl", "warehouse"}:
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

            # Enable UUID extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # Create tables
            for table in WAREHOUSE_SCHEMAS:
                ddl = table.create_ddl()
                logger.debug(f"Creating table {table.full_name}")
                await conn.execute(ddl)

            # Create indexes concurrently (outside transaction)
            for table in WAREHOUSE_SCHEMAS:
                for idx_ddl in table.index_ddl():
                    try:
                        await conn.execute(idx_ddl)
                    except asyncpg.DuplicateObjectError:
                        pass

            # Create analytical views
            for view_name, view_ddl in ANALYTICS_VIEWS.items():
                logger.debug(f"Creating view etl.{view_name}")
                await conn.execute(view_ddl)

        logger.info("Warehouse schema bootstrapped successfully")

    async def bulk_insert(
        self,
        table: str,
        df: pd.DataFrame,
        on_conflict: str = "DO NOTHING",
    ) -> int:
        """High-performance bulk insert using COPY protocol."""
        if df.empty:
            return 0

        columns = df.columns.tolist()
        records = [tuple(row) for row in df.itertuples(index=False)]

        async with self._pool.acquire() as conn:
            result = await conn.copy_records_to_table(
                table.split(".")[-1],
                schema_name=table.split(".")[0] if "." in table else "public",
                records=records,
                columns=columns,
            )

        loaded = len(records)
        logger.debug(f"Bulk inserted {loaded:,} records into {table}")
        return loaded

    async def execute_query(self, sql: str, *args) -> list[dict]:
        """Execute a SELECT query and return rows as dicts."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(row) for row in rows]

    async def get_pipeline_summary(self) -> pd.DataFrame:
        """Fetch the pipeline summary analytical view."""
        rows = await self.execute_query("SELECT * FROM etl.pipeline_summary;")
        return pd.DataFrame(rows)

    async def get_anomaly_trend(self, days: int = 30) -> pd.DataFrame:
        rows = await self.execute_query(
            """
            SELECT * FROM etl.anomaly_rate_trend
            WHERE day >= NOW() - INTERVAL '$1 days'
            ORDER BY day DESC;
            """,
            days,
        )
        return pd.DataFrame(rows)

    async def upsert_pipeline_run(self, run_data: dict) -> str:
        """Insert or update a pipeline run record."""
        sql = """
        INSERT INTO etl.pipeline_runs
            (pipeline_name, status, records_extracted, records_loaded,
             records_rejected, anomalies_detected, duration_seconds,
             throughput_rps, error_message, metadata, completed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, NOW())
        RETURNING run_id::text;
        """
        import json
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                run_data["pipeline_name"],
                run_data["status"],
                run_data.get("records_extracted", 0),
                run_data.get("records_loaded", 0),
                run_data.get("records_rejected", 0),
                run_data.get("anomalies_detected", 0),
                run_data.get("duration_seconds"),
                run_data.get("throughput_rps"),
                run_data.get("error_message"),
                json.dumps(run_data.get("metadata", {})),
            )
        return row["run_id"]
