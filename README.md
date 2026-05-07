# Neural ETL Pipeline

> **Production-grade ETL pipeline** with TensorFlow-powered anomaly detection, async batch processing, and SQL data warehousing.

[![CI/CD](https://github.com/yourusername/neural-etl-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/neural-etl-pipeline/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

Neural ETL Pipeline ingests data from heterogeneous sources (PostgreSQL, REST APIs, Parquet/CSV files), applies ML-powered quality checks, and loads clean records into a SQL data warehouse — all with async parallelism, retry logic, and full observability.

```
Sources              Extract          Transform           ML Filter          Load
─────────            ───────          ─────────           ─────────          ────
PostgreSQL  ─────►  Streaming  ──►  Validation  ──►  TF Autoencoder ──►  Warehouse
REST APIs           Batching         Type Cast          Anomaly Score       PostgreSQL
Parquet/CSV         Async I/O        Checksums          Quarantine          Bulk Insert
```

---

## Key Features

| Feature | Description |
|---|---|
| **TF Autoencoder** | Variational autoencoder detects data anomalies via reconstruction error |
| **Async Batch ETL** | `asyncio`-native pipeline with configurable parallelism and backpressure |
| **Multi-Source** | PostgreSQL, REST APIs, CSV, Parquet, JSON via pluggable connectors |
| **SQL Warehouse** | Schema management, COPY-protocol bulk inserts, analytical views |
| **Validation Rules** | Declarative per-field rules: `not_null`, `positive`, `is_email`, `regex`, `in_range` |
| **Deduplication** | SHA-256 row checksums prevent duplicate loads |
| **Checkpointing** | Resume interrupted pipelines from last committed offset |
| **Lifecycle Hooks** | Pre/post hooks for `extract`, `transform`, `load` stages |
| **Observability** | Structured logging, per-batch metrics, pipeline summary views |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeuralETLPipeline                            │
│                                                                     │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐   │
│  │ DataExtractor│──►│  DataTransformer │──►│   AnomalyDetector  │   │
│  │             │   │                  │   │                    │   │
│  │ SQLConnector│   │ ValidationRules   │   │  AutoencoderModel  │   │
│  │ APIConnector│   │ ColumnCasting     │   │  (TensorFlow)      │   │
│  │ FileConnector   │ SHA-256 Checksum  │   │  Reconstruction    │   │
│  └─────────────┘   └──────────────────┘   │  Error Scoring     │   │
│                                           └────────┬───────────┘   │
│                                                    │               │
│                    ┌───────────────────────────────▼───────────┐   │
│                    │           DataLoader                       │   │
│                    │  ┌───────────────────────────────────┐    │   │
│                    │  │        WarehouseManager            │    │   │
│                    │  │  COPY Protocol · asyncpg Pool      │    │   │
│                    │  │  etl.pipeline_runs                 │    │   │
│                    │  │  warehouse.data_records            │    │   │
│                    │  │  etl.anomaly_log                   │    │   │
│                    │  └───────────────────────────────────┘    │   │
│                    └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/yourusername/neural-etl-pipeline
cd neural-etl-pipeline
pip install -e ".[dev]"
```

### Run a Pipeline

```python
import asyncio
from src.etl.pipeline import NeuralETLPipeline, PipelineConfig

config = PipelineConfig(
    name="sales_pipeline",
    batch_size=10_000,
    enable_anomaly_detection=True,
    anomaly_threshold=0.95,
    parallel_workers=4,
)

pipeline = NeuralETLPipeline(config)

result = asyncio.run(pipeline.run(
    source="postgresql://user:pass@localhost/sales_db",
    destination="warehouse.data_records",
    filters={"active": True},
))

print(f"Loaded {result.records_loaded:,} records | {result.throughput:.0f} rec/s")
print(f"Anomalies detected: {result.anomalies_detected}")
```

### Configure Transformations

```python
from src.etl.transformer import DataTransformer, TransformationConfig, ValidationRule

config = TransformationConfig(
    rename_columns={"userId": "user_id", "txnAmt": "amount"},
    cast_columns={"amount": "float64", "created_at": "datetime"},
    fill_na={"category": "unknown", "score": 0.0},
    validation_rules=[
        ValidationRule("amount", "positive"),
        ValidationRule("email", "is_email"),
        ValidationRule("age", "in_range", params={"min": 0, "max": 120}),
        ValidationRule("status", "in_set", params={"values": ["active", "inactive"]}),
    ],
    add_checksum=True,
)
```

### Train the Anomaly Detector

```python
from src.models.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(
    threshold=0.95,   # flag top 5% reconstruction errors
    latent_dim=16,
    epochs=50,
)

# Train on clean reference data
stats = await detector.fit(clean_reference_df)
print(f"Threshold: {stats['threshold']:.6f}")

# Filter production data
clean, anomalies = await detector.filter(production_df)
print(f"Quarantined {len(anomalies)} anomalous records")

# Persist trained model
detector.save(Path("models/anomaly_detector"))
```

---

## SQL Schema

The warehouse includes four core tables and three analytical views:

### Tables

| Table | Schema | Description |
|---|---|---|
| `pipeline_runs` | `etl` | Execution history with metrics |
| `data_records` | `warehouse` | Processed records (range-partitioned) |
| `anomaly_log` | `etl` | Quarantined anomalous records |
| `batch_metrics` | `etl` | Per-batch performance data |

### Analytical Views

```sql
-- Overall pipeline health
SELECT * FROM etl.pipeline_summary;

-- Hourly throughput trend
SELECT * FROM etl.hourly_throughput WHERE hour > NOW() - INTERVAL '7 days';

-- Anomaly rate over time
SELECT * FROM etl.anomaly_rate_trend ORDER BY day DESC LIMIT 30;
```

---

## TensorFlow Model

The `AutoencoderModel` uses a symmetric encoder-decoder architecture:

```
Input (N features)
    │
    ▼
Dense(128, ReLU) + BatchNorm + Dropout(0.1)
    │
    ▼
Dense(64, ReLU)  + BatchNorm + Dropout(0.1)
    │
    ▼
Dense(32, ReLU)  + BatchNorm + Dropout(0.1)
    │
    ▼
  Latent Space (dim=16)
    │
    ▼  [mirror decoder]
    │
    ▼
Dense(N, Sigmoid)  ← Reconstruction
    │
    ▼
MSE(input, reconstruction) → Anomaly Score
```

Records where `score > percentile(train_errors, threshold)` are quarantined.

---

## Project Structure

```
neural-etl-pipeline/
├── src/
│   ├── etl/
│   │   ├── pipeline.py        # Core orchestrator
│   │   ├── extractor.py       # Multi-source streaming extractor
│   │   ├── transformer.py     # Validation & transformation engine
│   │   └── loader.py          # Bulk loader with deduplication
│   ├── models/
│   │   └── anomaly_detector.py # TF autoencoder anomaly detection
│   ├── sql/
│   │   └── warehouse.py       # Schema, DDL, analytical views
│   └── utils/
│       ├── logger.py          # Structured logging
│       └── metrics.py         # Pipeline metrics
├── tests/
│   └── test_pipeline.py       # 26-test suite
├── .github/workflows/ci.yml   # CI/CD (lint, test, docker, benchmark)
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Skip TF-dependent tests (no GPU/TF install needed)
pytest tests/ -k "not anomaly and not Integration"
```

---

## Tech Stack

- **Python 3.11+** — async/await, dataclasses, type hints
- **TensorFlow 2.15+** — Autoencoder anomaly detection
- **asyncpg** — Async PostgreSQL with COPY protocol
- **pandas / numpy** — Data processing
- **aiohttp** — Async REST API extraction
- **pyarrow** — Parquet streaming
- **pytest** — 26-test suite with fixtures
- **ruff + mypy** — Linting and static type checking
- **Docker + GitHub Actions** — CI/CD

---

## License

MIT © MIT © [Pradeep Kumar Kommuri](https://github.com/pradeepkumarkommuri)
