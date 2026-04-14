from __future__ import annotations

from typing import Protocol

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)


class PropertyDataProvider(Protocol):
    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        ...


class GeocodingProvider(Protocol):
    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        ...