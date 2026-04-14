from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from house_price_prediction.domain.contracts.prediction_contracts import FeatureVectorContract
from house_price_prediction.feature_schema import align_feature_payload


@dataclass(frozen=True)
class FeatureAssemblyService:
    model_name: str
    model_version: str
    expected_feature_names: tuple[str, ...]

    def assemble(
        self,
        request_id: UUID,
        provider_payload: dict[str, Any],
    ) -> FeatureVectorContract:
        aligned_features = align_feature_payload(self.expected_feature_names, provider_payload)
        total_features = len(aligned_features)
        non_null_features = sum(value is not None for value in aligned_features.values())
        completeness_score = 0.0
        if total_features:
            completeness_score = non_null_features / total_features

        return FeatureVectorContract(
            request_id=request_id,
            model_name=self.model_name,
            model_version=self.model_version,
            features=aligned_features,
            completeness_score=round(completeness_score, 4),
        )