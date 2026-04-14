from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_PREDICTION_FEATURES: tuple[str, ...] = (
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "Neighborhood",
    "HouseStyle",
)


def align_feature_payload(
    expected_feature_names: Iterable[str],
    source_features: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_feature_names = list(expected_feature_names)
    if not ordered_feature_names:
        return dict(source_features)
    return {feature_name: source_features.get(feature_name) for feature_name in ordered_feature_names}