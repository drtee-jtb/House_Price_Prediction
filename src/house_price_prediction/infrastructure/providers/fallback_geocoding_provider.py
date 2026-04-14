from __future__ import annotations

from dataclasses import dataclass

from house_price_prediction.domain.contracts.prediction_contracts import AddressPayload, GeocodingResultContract
from house_price_prediction.infrastructure.providers.base import GeocodingProvider


@dataclass(frozen=True)
class FallbackGeocodingProvider:
    providers: tuple[GeocodingProvider, ...]

    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return provider.normalize(address_payload)
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("No geocoding providers configured.")
        raise last_error