from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)


class HeuristicPropertyDataClient:
    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        postal_code = normalized_address.postal_code or "00000"
        latitude = normalized_address.latitude or 39.5
        longitude = normalized_address.longitude or -98.35
        seed = (
            f"{normalized_address.formatted_address}|{postal_code}|"
            f"{latitude:.4f}|{longitude:.4f}"
        )

        urban_score = self._fraction(seed, "urban_score")
        quality_score = self._fraction(seed, "quality_score")
        build_epoch = self._fraction(seed, "build_epoch")

        overall_qual = 5 + round(quality_score * 4)
        total_rooms = 5 + round(urban_score * 4)
        garage_cars = 1 + round((1 - urban_score) * 2)
        year_built = 1965 + round(build_epoch * 57)
        lot_area = 5000 + round((1 - urban_score) * 9000)

        payload = {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": 5 + round(self._fraction(seed, "condition") * 3),
            "YearBuilt": year_built,
            "YearRemodAdd": min(2024, year_built + 8 + round(self._fraction(seed, "remodel") * 12)),
            "GrLivArea": 1100 + (total_rooms * 220),
            "FullBath": 1 + round(self._fraction(seed, "full_bath") * 2),
            "HalfBath": round(self._fraction(seed, "half_bath")),
            "BedroomAbvGr": max(2, total_rooms // 2),
            "TotRmsAbvGrd": total_rooms,
            "Fireplaces": round(self._fraction(seed, "fireplaces") * 2),
            "GarageCars": garage_cars,
            "GarageArea": garage_cars * 240,
            "Neighborhood": self._neighborhood(normalized_address),
            "HouseStyle": "2Story" if urban_score >= 0.55 else "1Story",
            "feature_source": "heuristic",
            "feature_provenance": {
                "strategy": "heuristic",
                "providers": ["heuristic_property_data"],
                "derived_from": ["formatted_address", "postal_code", "coordinates"],
            },
        }

        return ProviderResponseContract(
            provider_name="heuristic_property_data",
            status="success",
            payload=payload,
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _fraction(seed: str, name: str) -> float:
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    @staticmethod
    def _neighborhood(normalized_address: NormalizedAddress) -> str:
        city = normalized_address.city.strip().upper()
        if city in {"WASHINGTON", "MIAMI", "MIAMI BEACH"}:
            return city.title().replace(" ", "")
        return city.title().replace(" ", "")[:12] or "Unknown"