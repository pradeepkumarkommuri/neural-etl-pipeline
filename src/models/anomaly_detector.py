"""
TensorFlow Autoencoder-based Anomaly Detector
Detects data anomalies in ETL pipelines using unsupervised learning.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutoencoderModel(keras.Model):
    """
    Variational Autoencoder for tabular anomaly detection.

    Architecture:
        Encoder: Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(latent_dim)
        Decoder: Dense(64, ReLU) → Dense(128, ReLU) → Dense(input_dim, Sigmoid)

    The reconstruction error is used as the anomaly score.
    Records exceeding the threshold are flagged as anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        encoder_dims: list[int] = None,
        dropout_rate: float = 0.1,
        l2_reg: float = 1e-4,
    ) -> None:
        super().__init__()
        encoder_dims = encoder_dims or [128, 64, 32]
        decoder_dims = list(reversed(encoder_dims))

        # Build encoder
        encoder_layers = []
        for i, dim in enumerate(encoder_dims):
            encoder_layers.append(
                layers.Dense(
                    dim,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f"encoder_{i}",
                )
            )
            encoder_layers.append(layers.BatchNormalization())
            encoder_layers.append(layers.Dropout(dropout_rate))

        encoder_layers.append(layers.Dense(latent_dim, name="latent_space"))
        self.encoder = keras.Sequential(encoder_layers, name="encoder")

        # Build decoder
        decoder_layers = []
        for i, dim in enumerate(decoder_dims):
            decoder_layers.append(
                layers.Dense(
                    dim,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f"decoder_{i}",
                )
            )
            decoder_layers.append(layers.BatchNormalization())
            decoder_layers.append(layers.Dropout(dropout_rate))

        decoder_layers.append(layers.Dense(input_dim, activation="sigmoid", name="reconstruction"))
        self.decoder = keras.Sequential(decoder_layers, name="decoder")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        latent = self.encoder(x, training=training)
        reconstructed = self.decoder(latent, training=training)
        return reconstructed

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder(x, training=False)


class AnomalyDetector:
    """
    Production anomaly detector for ETL pipelines.

    Uses a trained autoencoder to compute per-record reconstruction error.
    Records with error above the learned threshold are quarantined.

    Example:
        >>> detector = AnomalyDetector(threshold=0.95)
        >>> await detector.fit(training_data)
        >>> clean, anomalies = await detector.filter(new_data)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        latent_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 256,
        model_path: Optional[Path] = None,
    ) -> None:
        self.threshold = threshold
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.model: Optional[AutoencoderModel] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._error_threshold: Optional[float] = None
        self._is_fitted = False

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
        """Select numeric features and apply z-score normalization."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for anomaly detection")

        X = df[numeric_cols].fillna(0).values.astype(np.float32)

        if self._scaler_mean is None:
            self._scaler_mean = X.mean(axis=0)
            self._scaler_std = X.std(axis=0) + 1e-8

        X_norm = (X - self._scaler_mean) / self._scaler_std
        # Clip and rescale to [0, 1] for sigmoid output compatibility
        X_norm = np.clip(X_norm, -3, 3) / 6.0 + 0.5

        return X_norm.astype(np.float32), numeric_cols

    async def fit(self, df: pd.DataFrame, validation_split: float = 0.1) -> dict:
        """Train the autoencoder on clean reference data."""

        logger.info(f"Training anomaly detector on {len(df):,} records")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fit_sync, df, validation_split)

    def _fit_sync(self, df: pd.DataFrame, validation_split: float) -> dict:
        X, feature_cols = self._preprocess(df)
        input_dim = X.shape[1]

        self.model = AutoencoderModel(input_dim=input_dim, latent_dim=self.latent_dim)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        history = self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )

        # Compute threshold from training reconstruction errors
        reconstructed = self.model(tf.constant(X), training=False).numpy()
        errors = np.mean(np.square(X - reconstructed), axis=1)
        self._error_threshold = float(np.percentile(errors, self.threshold * 100))
        self._is_fitted = True

        logger.info(
            f"Detector trained | features={len(feature_cols)} | "
            f"latent_dim={self.latent_dim} | threshold={self._error_threshold:.6f} | "
            f"epochs={len(history.history['loss'])}"
        )

        if self.model_path:
            self.save(self.model_path)

        return {
            "input_dim": input_dim,
            "features": feature_cols,
            "threshold": self._error_threshold,
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [0])[-1]),
        }

    async def filter(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter DataFrame into clean records and anomalies.

        Returns:
            Tuple[clean_df, anomaly_df]
        """
        if not self._is_fitted:
            logger.warning("Detector not fitted — skipping anomaly detection")
            return df, pd.DataFrame()

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._score_sync, df)

        anomaly_mask = scores > self._error_threshold
        clean_df = df[~anomaly_mask].copy()
        anomaly_df = df[anomaly_mask].copy()
        anomaly_df["_anomaly_score"] = scores[anomaly_mask]

        logger.debug(
            f"Anomaly scan: total={len(df):,} | "
            f"clean={len(clean_df):,} | "
            f"anomalies={len(anomaly_df):,}"
        )
        return clean_df, anomaly_df

    def _score_sync(self, df: pd.DataFrame) -> np.ndarray:
        """Compute reconstruction error scores for each record."""
        X, _ = self._preprocess(df)
        X_tensor = tf.constant(X)
        reconstructed = self.model(X_tensor, training=False).numpy()
        return np.mean(np.square(X - reconstructed), axis=1)

    def save(self, path: Path) -> None:
        """Persist the model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(path / "weights.h5"))
        np.save(str(path / "scaler_mean.npy"), self._scaler_mean)
        np.save(str(path / "scaler_std.npy"), self._scaler_std)
        np.save(str(path / "error_threshold.npy"), np.array([self._error_threshold]))
        logger.info(f"Anomaly detector saved to {path}")

    def load(self, path: Path) -> None:
        """Restore model from disk."""
        path = Path(path)
        self._scaler_mean = np.load(str(path / "scaler_mean.npy"))
        self._scaler_std = np.load(str(path / "scaler_std.npy"))
        self._error_threshold = float(np.load(str(path / "error_threshold.npy"))[0])
        input_dim = len(self._scaler_mean)
        self.model = AutoencoderModel(input_dim=input_dim, latent_dim=self.latent_dim)
        self.model.build((None, input_dim))
        self.model.load_weights(str(path / "weights.h5"))
        self._is_fitted = True
        logger.info(f"Anomaly detector loaded from {path} | threshold={self._error_threshold:.6f}")
