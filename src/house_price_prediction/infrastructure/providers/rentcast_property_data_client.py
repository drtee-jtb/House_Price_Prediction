"""
Rentcast property data client.

Uses the Rentcast API (api.rentcast.io) to fetch real property records:
beds, baths, sqft, year built, lot size, property type.

Free tier: 50 requests/month — no credit card required.
Sign up at https://app.rentcast.io/app/api-access to get an API key,
then set RENTCAST_API_KEY in your environment.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError


class RentcastPropertyDataClient:
    """Fetch real property details from Rentcast API."""

    BASE_URL = "https://api.rentcast.io/v1"

    def __init__(
        self,
        api_key: str,
        fallback_provider: PropertyDataProvider | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("RENTCAST_API_KEY is required")
        self._api_key = api_key
        self._fallback_provider = fallback_provider

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        fallback_response = (
            self._fallback_provider.fetch_property_features(normalized_address)
            if self._fallback_provider is not None
            else None
        )

        try:
            data = self._fetch_property(normalized_address)
            payload = dict(fallback_response.payload) if fallback_response else {}
            payload.update(self._map_to_features(data, normalized_address))
            payload["feature_source"] = "rentcast"
            payload["feature_provenance"] = {
                "strategy": "rentcast_api",
                "providers": ["rentcast"],
                "derived_from": ["address"],
            }
            return ProviderResponseContract(
                provider_name="rentcast",
                status="success",
                payload=payload,
                fetched_at=datetime.now(UTC),
            )

        except NonRetryableProviderError:
            raise
        except Exception as exc:
            if fallback_response is not None:
                return fallback_response
            raise NonRetryableProviderError(f"Rentcast lookup failed: {exc}") from exc

    def _fetch_property(self, addr: NormalizedAddress) -> dict:
        """Call Rentcast /properties endpoint for a single address."""
        address_str = addr.address_line_1
        params = {
            "address": address_str,
            "city": addr.city,
            "state": addr.state,
            "zipCode": addr.postal_code,
        }
        resp = httpx.get(
            f"{self.BASE_URL}/properties",
            params=params,
            headers={
                "X-Api-Key": self._api_key,
                "Accept": "application/json",
            },
            timeout=10.0,
        )
        if resp.status_code == 401:
            raise NonRetryableProviderError("Invalid Rentcast API key")
        if resp.status_code == 429:
            raise NonRetryableProviderError("Rentcast rate limit exceeded (50/month free tier)")
        resp.raise_for_status()
        data = resp.json()
        # Response is a list when querying by address
        if isinstance(data, list):
            if not data:
                raise NonRetryableProviderError("Rentcast returned no results for this address")
            data = data[0]
        return data

    def _map_to_features(self, data: dict, addr: NormalizedAddress) -> dict:
        """Map Rentcast response fields to model feature names."""
        beds        = data.get("bedrooms")
        full_bath   = data.get("bathrooms")
        half_bath   = 0
        # Rentcast sometimes returns fractional bathrooms (e.g. 2.5 = 2 full + 1 half)
        if full_bath is not None and full_bath != int(full_bath):
            half_bath = 1
            full_bath = int(full_bath)

        sqft        = data.get("squareFootage")
        lot_sqft    = data.get("lotSize")          # already in sqft
        yr_built    = data.get("yearBuilt")
        garage      = data.get("garageSpaces") or data.get("garage")
        prop_type   = data.get("propertyType", "Single Family")

        features: dict = {}

        if beds is not None:
            features["BedroomAbvGr"] = int(beds)
        if full_bath is not None:
            features["FullBath"] = int(full_bath)
            features["HalfBath"] = int(half_bath)
        if sqft is not None:
            features["GrLivArea"] = float(sqft)
        if lot_sqft is not None:
            features["LotArea"] = float(lot_sqft)
        if yr_built is not None:
            features["YearBuilt"] = int(yr_built)
            features["YearRemodAdd"] = int(yr_built) + 10
        if garage is not None:
            gc = int(garage)
            features["GarageCars"] = gc
            features["GarageArea"] = gc * 240

        # Derived
        _beds     = features.get("BedroomAbvGr", 3)
        _fbath    = features.get("FullBath", 1)
        features["TotRmsAbvGrd"] = _beds + _fbath + 2

        # Normalize property type label
        _type_map = {
            "Single Family": "single_family",
            "Condo": "condo",
            "Townhouse": "townhouse",
            "Multi Family": "multi_family",
        }
        features["PropertyType"] = _type_map.get(prop_type, "single_family")
        features["City"]  = addr.city
        features["State"] = addr.state

        return features
