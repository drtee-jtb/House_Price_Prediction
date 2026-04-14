from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)


class NominatimGeocodingClient:
    def __init__(self, base_url: str = "https://nominatim.openstreetmap.org") -> None:
        self._base_url = base_url.rstrip("/")

    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        address_parts = [
            address_payload.address_line_1,
            address_payload.city,
            address_payload.state,
            address_payload.postal_code,
            address_payload.country,
        ]
        query = ", ".join(part.strip() for part in address_parts if part)

        response = httpx.get(
            f"{self._base_url}/search",
            params={
                "q": query,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 1,
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            raise RuntimeError("Nominatim returned no address matches.")

        best_match = results[0]
        address = best_match.get("address", {})

        address_line_1 = " ".join(address_payload.address_line_1.strip().upper().split())
        city = " ".join(
            str(address.get("city") or address.get("town") or address.get("village") or address_payload.city)
            .strip()
            .upper()
            .split()
        )
        state = str(address.get("state") or address_payload.state).strip().upper()
        postal_code = str(address.get("postcode") or address_payload.postal_code).strip().upper()
        country = str(address.get("country_code") or address_payload.country).strip().upper()
        address_line_2 = (
            " ".join(address_payload.address_line_2.strip().upper().split())
            if address_payload.address_line_2
            else None
        )

        normalized_address = NormalizedAddress(
            address_line_1=address_line_1,
            address_line_2=address_line_2,
            city=city,
            state=state,
            postal_code=postal_code,
            country=country,
            formatted_address=(
                f"{address_line_1}, {city}, {state} {postal_code}, {country}"
            ),
            latitude=float(best_match["lat"]),
            longitude=float(best_match["lon"]),
            geocoding_source="nominatim",
        )
        return GeocodingResultContract(
            normalized_address=normalized_address,
            provider_response=ProviderResponseContract(
                provider_name="nominatim_geocoding",
                status="success",
                payload={"result": best_match},
                fetched_at=datetime.now(UTC),
            ),
        )