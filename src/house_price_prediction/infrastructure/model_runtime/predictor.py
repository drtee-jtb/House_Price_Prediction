from __future__ import annotations

from typing import Any

import pandas as pd

from house_price_prediction.config import Settings
from house_price_prediction.feature_schema import DEFAULT_PREDICTION_FEATURES
from house_price_prediction.model import TrainedModelArtifact, load_model_artifact


class ModelNotReadyError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


class PredictionRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._artifact: TrainedModelArtifact | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def model_version(self) -> str:
        return self._settings.model_version

    def is_available(self) -> bool:
        return self._settings.enable_mock_predictor or self._settings.model_path.exists()

    def expected_feature_names(self) -> tuple[str, ...]:
        if self._settings.model_path.exists():
            metadata = self._load_artifact().metadata
            if metadata.feature_columns:
                return metadata.feature_columns
        return DEFAULT_PREDICTION_FEATURES

    def predict(self, features: dict[str, Any]) -> float:
        if self._settings.enable_mock_predictor:
            return self._mock_predict(features)

        if not self._settings.model_path.exists():
            raise ModelNotReadyError(
                f"Model file not found at {self._settings.model_path}. Train the model or enable the mock predictor."
            )

        model = self._load_artifact().model
        try:
            prediction = model.predict(pd.DataFrame([features]))
        except Exception as exc:
            raise ModelInferenceError(f"Prediction failed: {exc}") from exc
        return float(prediction[0])

    def _load_artifact(self) -> TrainedModelArtifact:
        if self._artifact is None:
            self._artifact = load_model_artifact(self._settings.model_path)
        return self._artifact

    @staticmethod
    def _mock_predict(features: dict[str, Any]) -> float:
        lot_area = float(features.get("LotArea", 8500))
        overall_qual = float(features.get("OverallQual", 6))
        gr_liv_area = float(features.get("GrLivArea", 1500))
        garage_cars = float(features.get("GarageCars", 2))
        full_bath = float(features.get("FullBath", 2))
        fireplaces = float(features.get("Fireplaces", 1))
        year_built = float(features.get("YearBuilt", 1995))

        age_premium = max(year_built - 1950, 0) * 700
        predicted_price = (
            50000
            + (lot_area * 2.3)
            + (overall_qual * 18000)
            + (gr_liv_area * 92)
            + (garage_cars * 9500)
            + (full_bath * 6500)
            + (fireplaces * 4500)
            + age_premium
        )
        return round(predicted_price, 2)