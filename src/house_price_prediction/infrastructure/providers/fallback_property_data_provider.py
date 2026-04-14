from __future__ import annotations

from dataclasses import dataclass

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider


@dataclass(frozen=True)
class FallbackPropertyDataProvider:
    providers: tuple[PropertyDataProvider, ...]

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return provider.fetch_property_features(normalized_address)
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("No property data providers configured.")
        raise last_error