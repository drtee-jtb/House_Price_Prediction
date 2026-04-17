from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from house_price_prediction.application.services.feature_policy_registry import (
    get_feature_policy_weights,
    list_feature_policy_names,
)
from house_price_prediction.domain.contracts.prediction_contracts import FeatureVectorContract
from house_price_prediction.feature_schema import align_feature_payload


@dataclass(frozen=True)
class FeatureAssemblyService:
    model_name: str
    model_version: str
    expected_feature_names: tuple[str, ...]
    feature_policy_name: str = "balanced-v1"
    feature_policy_version: str = "v1"
    feature_policy_state_overrides: dict[str, str] = field(default_factory=dict)

    def assemble(
        self,
        request_id: UUID,
        provider_payload: dict[str, Any],
        context: dict[str, Any] | None = None,
        policy_name_override: str | None = None,
    ) -> FeatureVectorContract:
        aligned_features = align_feature_payload(self.expected_feature_names, provider_payload)
        selected_policy_name = policy_name_override or self.resolve_policy_for_context(context)
        feature_weights = self._resolve_feature_weights(aligned_features, selected_policy_name)
        total_weight = sum(feature_weights.values())
        non_null_weight = sum(
            feature_weights[name]
            for name, value in aligned_features.items()
            if value is not None
        )
        completeness_score = 0.0
        if total_weight > 0:
            completeness_score = non_null_weight / total_weight

        return FeatureVectorContract(
            request_id=request_id,
            model_name=self.model_name,
            model_version=self.model_version,
            features=aligned_features,
            completeness_score=round(completeness_score, 4),
            feature_policy_name=selected_policy_name,
            feature_policy_version=self.feature_policy_version,
            weight_total=round(total_weight, 4),
        )

    def resolve_policy_for_context(self, context: dict[str, Any] | None = None) -> str:
        if context is not None:
            state = context.get("state")
            if isinstance(state, str):
                override = self.feature_policy_state_overrides.get(state.strip().upper())
                if override:
                    return override
        return self.feature_policy_name

    @staticmethod
    def available_policy_names() -> tuple[str, ...]:
        return list_feature_policy_names()

    def _resolve_feature_weights(
        self,
        aligned_features: dict[str, Any],
        selected_policy_name: str,
    ) -> dict[str, float]:
        profile = get_feature_policy_weights(selected_policy_name)

        return {
            name: max(float(profile.get(name, 1.0)), 0.0)
            for name in aligned_features
        }