from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError

# Nominatim returns full state names (e.g. "Alabama") but all downstream
# logic — feature policy state overrides, Census FIPS lookups, and DB storage —
# expects standard 2-letter USPS abbreviations ("AL").
_US_STATE_ABBREV: dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC", "WASHINGTON DC": "DC",
    "PUERTO RICO": "PR", "GUAM": "GU", "VIRGIN ISLANDS": "VI",
    "AMERICAN SAMOA": "AS", "NORTHERN MARIANA ISLANDS": "MP",
}


def _normalize_state(raw_state: str) -> str:
    """Convert a US state name or abbreviation to a 2-letter USPS abbreviation.

    Nominatim returns full state names ("Alabama"); Census geocoder returns
    FIPS numeric codes; callers may pass 2-letter abbreviations directly.
    This function normalises all three forms to the standard 2-letter code.
    If the value is already a 2-letter abbreviation it is returned as-is
    (uppercased).  Unknown values are returned uppercased without conversion
    so they remain visible in logs and traces.
    """
    cleaned = raw_state.strip().upper()
    if len(cleaned) == 2:
        return cleaned  # already an abbreviation
    return _US_STATE_ABBREV.get(cleaned, cleaned)


class NominatimGeocodingClient:
    def __init__(self, base_url: str = "https://nominatim.openstreetmap.org") -> None:
        self._base_url = base_url.rstrip("/")

    _HTTPX_TIMEOUT: float = 10.0

    def _search(self, query: str) -> list[dict]:
        response = httpx.get(
            f"{self._base_url}/search",
            params={
                "q": query,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 1,
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
            timeout=self._HTTPX_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        full_parts = [
            address_payload.address_line_1,
            address_payload.city,
            address_payload.state,
            address_payload.postal_code,
            address_payload.country,
        ]
        full_query = ", ".join(part.strip() for part in full_parts if part)
        results = self._search(full_query)

        # Fallback: drop postal code and retry with city/state/country only to
        # get at least a city-centroid coordinate when the full query yields nothing.
        geocoding_source = "nominatim"
        if not results:
            fallback_parts = [
                address_payload.city,
                address_payload.state,
                address_payload.country,
            ]
            fallback_query = ", ".join(part.strip() for part in fallback_parts if part)
            results = self._search(fallback_query)
            geocoding_source = "nominatim_city_fallback"

        if not results:
            raise NonRetryableProviderError("Nominatim returned no address matches.")

        best_match = results[0]
        address = best_match.get("address", {})

        address_line_1 = " ".join(address_payload.address_line_1.strip().upper().split())
        city = " ".join(
            str(address.get("city") or address.get("town") or address.get("village") or address_payload.city)
            .strip()
            .upper()
            .split()
        )
        state = _normalize_state(str(address.get("state") or address_payload.state))
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
            geocoding_source=geocoding_source,
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