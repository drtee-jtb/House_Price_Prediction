from __future__ import annotations

from dataclasses import dataclass

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider


@dataclass(frozen=True)
class PropertyEnrichmentService:
    property_data_client: PropertyDataProvider

    def build_property_record(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        return self.property_data_client.fetch_property_features(normalized_address)