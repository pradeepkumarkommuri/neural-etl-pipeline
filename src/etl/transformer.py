"""
Data Transformer — Schema validation, cleaning, and enrichment
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """Declarative field validation rule."""
    field: str
    rule: str
    params: dict = field(default_factory=dict)
    severity: str = "error"  # "error" | "warning"
    message: Optional[str] = None

    BUILT_IN_RULES = {
        "not_null": lambda s, _: ~s.isna(),
        "positive": lambda s, _: s > 0,
        "in_range": lambda s, p: (s >= p["min"]) & (s <= p["max"]),
        "regex": lambda s, p: s.astype(str).str.match(p["pattern"]),
        "max_length": lambda s, p: s.astype(str).str.len() <= p["length"],
        "unique": lambda s, _: ~s.duplicated(keep="first"),
        "in_set": lambda s, p: s.isin(p["values"]),
        "is_email": lambda s, _: s.astype(str).str.match(
            r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        ),
        "is_date": lambda s, _: pd.to_datetime(s, errors="coerce").notna(),
    }

    def validate(self, series: pd.Series) -> pd.Series:
        """Return boolean mask where True = valid."""
        if self.rule not in self.BUILT_IN_RULES:
            raise ValueError(f"Unknown validation rule: '{self.rule}'")
        try:
            return self.BUILT_IN_RULES[self.rule](series, self.params)
        except Exception:
            return pd.Series([True] * len(series), index=series.index)


@dataclass
class TransformationConfig:
    """Configuration for data transformation pipeline."""
    rename_columns: dict[str, str] = field(default_factory=dict)
    drop_columns: list[str] = field(default_factory=list)
    cast_columns: dict[str, str] = field(default_factory=dict)  # col: dtype
    fill_na: dict[str, Any] = field(default_factory=dict)
    validation_rules: list[ValidationRule] = field(default_factory=list)
    add_checksum: bool = True
    add_ingestion_timestamp: bool = True
    custom_transforms: list[Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=list
    )


class DataTransformer:
    """
    Async data transformer with declarative rules and validation.

    Features:
    - Column renaming, dropping, type casting
    - NULL filling strategies
    - Declarative field validation with severity levels
    - SHA-256 row checksums for deduplication
    - Ingestion timestamp injection
    - Custom transform function hooks

    Example:
        >>> config = TransformationConfig(
        ...     rename_columns={"userId": "user_id"},
        ...     cast_columns={"amount": "float64"},
        ...     validation_rules=[
        ...         ValidationRule("amount", "positive"),
        ...         ValidationRule("email", "is_email"),
        ...     ]
        ... )
        >>> transformer = DataTransformer(config)
        >>> clean, rejected = await transformer.transform(df)
    """

    def __init__(self, config: Optional[TransformationConfig] = None) -> None:
        self.config = config or TransformationConfig()

    async def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply all transformations. Returns (clean_df, rejected_df).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transform_sync, df.copy())

    def _transform_sync(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config

        # 1. Drop columns
        cols_to_drop = [c for c in cfg.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 2. Rename columns
        df = df.rename(columns=cfg.rename_columns)

        # 3. Fill NAs
        for col, fill_val in cfg.fill_na.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        # 4. Type casting
        for col, dtype in cfg.cast_columns.items():
            if col in df.columns:
                try:
                    if dtype in ("datetime", "timestamp"):
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(dtype, errors="ignore")
                except Exception as e:
                    logger.warning(f"Failed to cast column '{col}' to {dtype}: {e}")

        # 5. Custom transforms
        for fn in cfg.custom_transforms:
            df = fn(df)

        # 6. Validation
        reject_mask = pd.Series([False] * len(df), index=df.index)
        violation_log = []

        for rule in cfg.validation_rules:
            if rule.field not in df.columns:
                logger.warning(f"Validation rule references missing column: '{rule.field}'")
                continue

            valid_mask = rule.validate(df[rule.field])
            failed = ~valid_mask
            failed_count = failed.sum()

            if failed_count > 0:
                msg = rule.message or f"Field '{rule.field}' failed rule '{rule.rule}'"
                violation_log.append(f"{msg}: {failed_count} records")

                if rule.severity == "error":
                    reject_mask |= failed
                else:
                    logger.warning(f"[WARNING] {msg} in {failed_count} records")

        if violation_log:
            logger.info(f"Validation violations: {'; '.join(violation_log)}")

        rejected_df = df[reject_mask].copy()
        clean_df = df[~reject_mask].copy()

        # 7. Add metadata columns
        if cfg.add_ingestion_timestamp:
            clean_df["_ingested_at"] = datetime.utcnow()

        if cfg.add_checksum:
            clean_df["_checksum"] = self._compute_checksums(clean_df)

        logger.debug(
            f"Transform complete | in={len(df):,} | "
            f"clean={len(clean_df):,} | rejected={len(rejected_df):,}"
        )
        return clean_df, rejected_df

    @staticmethod
    def _compute_checksums(df: pd.DataFrame) -> pd.Series:
        """SHA-256 checksum per row for deduplication detection."""
        meta_cols = [c for c in df.columns if c.startswith("_")]
        data_df = df.drop(columns=meta_cols, errors="ignore")

        def row_hash(row):
            row_str = json.dumps(row.to_dict(), sort_keys=True, default=str)
            return hashlib.sha256(row_str.encode()).hexdigest()[:16]

        return data_df.apply(row_hash, axis=1)

    def add_custom_transform(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Dynamically add a custom transformation function."""
        self.config.custom_transforms.append(fn)
        logger.debug(f"Custom transform added: {fn.__name__}")
