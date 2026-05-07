"""
Test suite for Neural ETL Pipeline
Tests: transformer, anomaly detector, SQL schema, pipeline orchestration
"""

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Standard clean test DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "user_id": range(1, 101),
        "amount": np.random.uniform(10, 500, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "email": [f"user{i}@example.com" for i in range(100)],
        "active": np.random.choice([True, False], 100),
        "score": np.random.uniform(0, 1, 100),
    })


@pytest.fixture
def dirty_df(sample_df):
    """DataFrame with nulls, bad types, and outliers injected."""
    df = sample_df.copy()
    df.loc[[5, 15, 25], "amount"] = None
    df.loc[[10, 20], "email"] = "not-an-email"
    df.loc[[30], "amount"] = -999.0  # outlier
    return df


# ── Transformer Tests ──────────────────────────────────────────────────────

class TestDataTransformer:

    def test_import(self):
        from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule
        assert DataTransformer is not None

    def test_basic_transform(self, sample_df):
        from src.etl.transformer import DataTransformer, TransformationConfig
        config = TransformationConfig(add_checksum=True, add_ingestion_timestamp=True)
        transformer = DataTransformer(config)
        clean, rejected = asyncio.get_event_loop().run_until_complete(
            transformer.transform(sample_df)
        )
        assert len(clean) == len(sample_df)
        assert len(rejected) == 0
        assert "_checksum" in clean.columns
        assert "_ingested_at" in clean.columns

    def test_rename_columns(self, sample_df):
        from src.etl.transformer import DataTransformer, TransformationConfig
        config = TransformationConfig(
            rename_columns={"user_id": "uid", "amount": "value"},
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        clean, _ = asyncio.get_event_loop().run_until_complete(
            transformer.transform(sample_df)
        )
        assert "uid" in clean.columns
        assert "value" in clean.columns
        assert "user_id" not in clean.columns

    def test_drop_columns(self, sample_df):
        from src.etl.transformer import DataTransformer, TransformationConfig
        config = TransformationConfig(
            drop_columns=["active", "score"],
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        clean, _ = asyncio.get_event_loop().run_until_complete(
            transformer.transform(sample_df)
        )
        assert "active" not in clean.columns
        assert "score" not in clean.columns

    def test_not_null_validation_rejects(self, dirty_df):
        from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule
        config = TransformationConfig(
            validation_rules=[ValidationRule("amount", "not_null")],
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        clean, rejected = asyncio.get_event_loop().run_until_complete(
            transformer.transform(dirty_df)
        )
        assert len(rejected) == 3  # 3 nulls injected
        assert len(clean) == len(dirty_df) - 3

    def test_email_validation(self, dirty_df):
        from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule
        config = TransformationConfig(
            validation_rules=[ValidationRule("email", "is_email")],
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        clean, rejected = asyncio.get_event_loop().run_until_complete(
            transformer.transform(dirty_df)
        )
        assert len(rejected) == 2  # 2 bad emails

    def test_positive_validation(self, dirty_df):
        from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule
        config = TransformationConfig(
            fill_na={"amount": 0},
            validation_rules=[ValidationRule("amount", "positive")],
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        clean, rejected = asyncio.get_event_loop().run_until_complete(
            transformer.transform(dirty_df)
        )
        # -999 outlier + 3 filled zeros = 4 rejected
        assert len(rejected) >= 1

    def test_checksum_deterministic(self, sample_df):
        from src.etl.transformer import DataTransformer, TransformationConfig
        config = TransformationConfig(add_checksum=True, add_ingestion_timestamp=False)
        t = DataTransformer(config)
        loop = asyncio.get_event_loop()
        clean1, _ = loop.run_until_complete(t.transform(sample_df))
        clean2, _ = loop.run_until_complete(t.transform(sample_df))
        assert (clean1["_checksum"] == clean2["_checksum"]).all()

    def test_fill_na(self):
        from src.etl.transformer import DataTransformer, TransformationConfig
        df = pd.DataFrame({"val": [1.0, None, 3.0], "name": [None, "b", "c"]})
        config = TransformationConfig(
            fill_na={"val": 0.0, "name": "unknown"},
            add_checksum=False,
            add_ingestion_timestamp=False,
        )
        t = DataTransformer(config)
        clean, _ = asyncio.get_event_loop().run_until_complete(t.transform(df))
        assert clean["val"].isna().sum() == 0
        assert clean["name"].isna().sum() == 0
        assert clean.loc[1, "val"] == 0.0
        assert clean.loc[0, "name"] == "unknown"

    def test_custom_transform_hook(self, sample_df):
        from src.etl.transformer import DataTransformer, TransformationConfig
        config = TransformationConfig(add_checksum=False, add_ingestion_timestamp=False)
        t = DataTransformer(config)
        t.add_custom_transform(lambda df: df.assign(doubled_score=df["score"] * 2))
        clean, _ = asyncio.get_event_loop().run_until_complete(t.transform(sample_df))
        assert "doubled_score" in clean.columns
        assert (clean["doubled_score"] == clean["score"] * 2).all()


# ── Anomaly Detector Tests ─────────────────────────────────────────────────

class TestAnomalyDetector:

    def _make_numeric_df(self, n=500, seed=0):
        np.random.seed(seed)
        return pd.DataFrame({
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(5, 2, n),
            "f3": np.random.uniform(0, 10, n),
        })

    def test_import(self):
        from src.models.anomaly_detector import AnomalyDetector
        assert AnomalyDetector is not None

    def test_detector_not_fitted_passthrough(self, sample_df):
        from src.models.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(threshold=0.95)
        loop = asyncio.get_event_loop()
        clean, anomalies = loop.run_until_complete(detector.filter(sample_df))
        # Not fitted → passthrough
        assert len(clean) == len(sample_df)
        assert len(anomalies) == 0

    def test_fit_and_filter(self):
        """Train on normal data, inject outliers, verify detection."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from src.models.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(threshold=0.95, epochs=5)
        normal = self._make_numeric_df(500)

        loop = asyncio.get_event_loop()
        stats = loop.run_until_complete(detector.fit(normal))
        assert stats["threshold"] > 0
        assert detector._is_fitted

        # Inject extreme outliers
        outliers = pd.DataFrame({
            "f1": [1000.0, -1000.0, 999.0],
            "f2": [500.0, -500.0, 400.0],
            "f3": [999.0, -999.0, 888.0],
        })
        test_data = pd.concat([normal, outliers], ignore_index=True)
        clean, anomalies = loop.run_until_complete(detector.filter(test_data))

        assert len(anomalies) > 0, "Expected outliers to be detected"
        assert "_anomaly_score" in anomalies.columns

    def test_autoencoder_architecture(self):
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from src.models.anomaly_detector import AutoencoderModel
        model = AutoencoderModel(input_dim=10, latent_dim=4)
        import numpy as np
        x = tf.constant(np.random.random((32, 10)).astype(np.float32))
        output = model(x, training=False)
        assert output.shape == (32, 10)

        latent = model.encode(x)
        assert latent.shape == (32, 4)


# ── SQL Schema Tests ───────────────────────────────────────────────────────

class TestSQLSchema:

    def test_import(self):
        from src.sql.warehouse import WarehouseManager, WAREHOUSE_SCHEMAS, TableSchema
        assert len(WAREHOUSE_SCHEMAS) == 4

    def test_table_ddl_generation(self):
        from src.sql.warehouse import WAREHOUSE_SCHEMAS
        for schema in WAREHOUSE_SCHEMAS:
            ddl = schema.create_ddl()
            assert "CREATE TABLE IF NOT EXISTS" in ddl
            assert schema.name in ddl

    def test_index_ddl_generation(self):
        from src.sql.warehouse import WAREHOUSE_SCHEMAS
        all_indexes = []
        for schema in WAREHOUSE_SCHEMAS:
            all_indexes.extend(schema.index_ddl())
        assert len(all_indexes) > 0
        for idx in all_indexes:
            assert "CREATE INDEX CONCURRENTLY" in idx

    def test_full_name_property(self):
        from src.sql.warehouse import WAREHOUSE_SCHEMAS
        for schema in WAREHOUSE_SCHEMAS:
            assert "." in schema.full_name

    def test_analytics_views_defined(self):
        from src.sql.warehouse import ANALYTICS_VIEWS
        assert "pipeline_summary" in ANALYTICS_VIEWS
        assert "hourly_throughput" in ANALYTICS_VIEWS
        assert "anomaly_rate_trend" in ANALYTICS_VIEWS
        for name, ddl in ANALYTICS_VIEWS.items():
            assert "CREATE OR REPLACE VIEW" in ddl


# ── Pipeline Orchestration Tests ───────────────────────────────────────────

class TestNeuralETLPipeline:

    def test_import(self):
        from src.etl.pipeline import NeuralETLPipeline, PipelineConfig, PipelineStatus
        assert NeuralETLPipeline is not None

    def test_pipeline_config_defaults(self):
        from src.etl.pipeline import PipelineConfig
        cfg = PipelineConfig(name="test")
        assert cfg.batch_size == 10_000
        assert cfg.max_retries == 3
        assert cfg.enable_anomaly_detection is True

    def test_pipeline_result_properties(self):
        from src.etl.pipeline import PipelineResult, PipelineStatus
        r = PipelineResult(
            pipeline_name="test",
            status=PipelineStatus.COMPLETED,
            records_extracted=1000,
            records_loaded=950,
            duration_seconds=10.0,
        )
        assert abs(r.success_rate - 95.0) < 0.01
        assert abs(r.throughput - 95.0) < 0.01

    def test_pipeline_result_zero_division(self):
        from src.etl.pipeline import PipelineResult, PipelineStatus
        r = PipelineResult(pipeline_name="test", status=PipelineStatus.IDLE)
        assert r.success_rate == 0.0
        assert r.throughput == 0.0

    def test_hook_registration(self):
        from src.etl.pipeline import NeuralETLPipeline, PipelineConfig
        cfg = PipelineConfig(name="test", enable_anomaly_detection=False, enable_metrics=False)
        pipeline = NeuralETLPipeline(cfg)
        callback = lambda x: x
        pipeline.register_hook("pre_extract", callback)
        assert callback in pipeline._hooks["pre_extract"]

    def test_invalid_hook_raises(self):
        from src.etl.pipeline import NeuralETLPipeline, PipelineConfig
        cfg = PipelineConfig(name="test", enable_anomaly_detection=False, enable_metrics=False)
        pipeline = NeuralETLPipeline(cfg)
        with pytest.raises(ValueError, match="Unknown event"):
            pipeline.register_hook("invalid_event", lambda x: x)


# ── Integration: Transform + Detect ───────────────────────────────────────

class TestIntegration:

    def test_transform_then_detect_pipeline(self):
        """Full transform → anomaly detect workflow without DB."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule
        from src.models.anomaly_detector import AnomalyDetector

        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "value": np.random.normal(50, 10, n),
            "count": np.random.randint(1, 100, n),
            "rate": np.random.uniform(0, 1, n),
            "email": [f"u{i}@test.com" for i in range(n)],
        })

        config = TransformationConfig(
            validation_rules=[ValidationRule("value", "positive")],
            add_checksum=True,
            add_ingestion_timestamp=False,
        )
        transformer = DataTransformer(config)
        loop = asyncio.get_event_loop()
        clean, rejected = loop.run_until_complete(transformer.transform(df))

        assert len(clean) > 0
        assert "_checksum" in clean.columns

        detector = AnomalyDetector(threshold=0.95, epochs=3)
        loop.run_until_complete(detector.fit(clean))

        normal_clean, normal_anomalies = loop.run_until_complete(detector.filter(clean))
        assert len(normal_clean) > len(normal_anomalies)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
