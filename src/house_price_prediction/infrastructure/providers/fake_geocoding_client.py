from __future__ import annotations

from datetime import UTC, datetime

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)


class FakeGeocodingClient:
    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        normalized_line_1 = " ".join(address_payload.address_line_1.strip().upper().split())
        normalized_city = " ".join(address_payload.city.strip().upper().split())
        normalized_state = address_payload.state.strip().upper()
        normalized_postal_code = address_payload.postal_code.strip().upper()
        normalized_country = address_payload.country.strip().upper()
        normalized_line_2 = (
            " ".join(address_payload.address_line_2.strip().upper().split())
            if address_payload.address_line_2
            else None
        )

        normalized_address = NormalizedAddress(
            address_line_1=normalized_line_1,
            address_line_2=normalized_line_2,
            city=normalized_city,
            state=normalized_state,
            postal_code=normalized_postal_code,
            country=normalized_country,
            formatted_address=(
                f"{normalized_line_1}, {normalized_city}, {normalized_state} "
                f"{normalized_postal_code}, {normalized_country}"
            ),
            geocoding_source="fake",
        )
        return GeocodingResultContract(
            normalized_address=normalized_address,
            provider_response=ProviderResponseContract(
                provider_name="fake_geocoding",
                status="success",
                payload={
                    "query": {
                        "address_line_1": address_payload.address_line_1,
                        "address_line_2": address_payload.address_line_2,
                        "city": address_payload.city,
                        "state": address_payload.state,
                        "postal_code": address_payload.postal_code,
                        "country": address_payload.country,
                    },
                    "normalized": normalized_address.model_dump(),
                },
                fetched_at=datetime.now(UTC),
            ),
        )