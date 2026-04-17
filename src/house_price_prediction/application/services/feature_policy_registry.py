from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePolicyDefinition:
    name: str
    version: str
    description: str
    emphasis_features: tuple[str, ...]
    weights: dict[str, float]


_FEATURE_POLICY_DEFINITIONS: tuple[FeaturePolicyDefinition, ...] = (
    FeaturePolicyDefinition(
        name="balanced-v1",
        version="v1",
        description="Uniform baseline weighting across all aligned model features.",
        emphasis_features=(),
        weights={},
    ),
    FeaturePolicyDefinition(
        name="quality-first-v1",
        version="v1",
        description="Emphasize structural quality and livable area signals for pricing.",
        emphasis_features=("OverallQual", "Neighborhood", "GrLivArea", "YearBuilt"),
        weights={
            "OverallQual": 3.0,
            "Neighborhood": 2.0,
            "GrLivArea": 2.0,
            "YearBuilt": 1.5,
        },
    ),
    FeaturePolicyDefinition(
        name="land-first-v1",
        version="v1",
        description="Emphasize lot size and land-adjacent characteristics.",
        emphasis_features=("LotArea", "LotFrontage", "Neighborhood"),
        weights={
            "LotArea": 3.0,
            "LotFrontage": 2.0,
            "Neighborhood": 1.5,
        },
    ),
)


def list_feature_policy_definitions() -> tuple[FeaturePolicyDefinition, ...]:
    return _FEATURE_POLICY_DEFINITIONS


def list_feature_policy_names() -> tuple[str, ...]:
    return tuple(policy.name for policy in _FEATURE_POLICY_DEFINITIONS)


def get_feature_policy_definition(policy_name: str) -> FeaturePolicyDefinition | None:
    normalized_name = policy_name.strip().lower()
    for policy in _FEATURE_POLICY_DEFINITIONS:
        if policy.name == normalized_name:
            return policy
    return None


def get_feature_policy_weights(policy_name: str) -> dict[str, float]:
    policy = get_feature_policy_definition(policy_name)
    if policy is None:
        logger.warning(
            "Unknown feature policy %r; falling back to 'balanced-v1'. "
            "Check FEATURE_POLICY_NAME or FEATURE_POLICY_STATE_OVERRIDES.",
            policy_name,
        )
        fallback = get_feature_policy_definition("balanced-v1")
        assert fallback is not None
        return fallback.weights
    return policy.weights
