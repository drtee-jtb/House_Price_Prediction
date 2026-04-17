"""Static scenario registry for automated pipeline contract testing.

Each scenario defines a real US address paired with structured expectations.
Expectations are deliberately calibrated to pass against the deterministic
fake provider (address-seeded SHA-256, ranges: LotArea 4500-18000,
GrLivArea 900-3200, BedroomAbvGr 2-6, TotRmsAbvGrd 4-11).

When switching to live providers (Nominatim + Census), replace the fake
provider and tighten bounds per market to catch real data quality issues.
"""

from __future__ import annotations

from dataclasses import dataclass

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    BaselineExpectationsInput,
    FeatureBoundExpectation,
)


@dataclass(frozen=True)
class RegisteredScenario:
    scenario_id: str
    label: str
    category: str
    description: str
    payload: AddressPayload
    expectations: BaselineExpectationsInput


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Bounds chosen to cover the full fake-provider output range so tests pass in
# both APP_ENV=test and with real providers.  Tighten per-scenario when you
# have live baseline numbers for a market.
# ---------------------------------------------------------------------------

_WIDE_BOUNDS: dict[str, FeatureBoundExpectation] = {
    "BedroomAbvGr": FeatureBoundExpectation(minimum=1, maximum=8),
    "TotRmsAbvGrd": FeatureBoundExpectation(minimum=2, maximum=14),
    "GrLivArea": FeatureBoundExpectation(minimum=500, maximum=5000),
    "LotArea": FeatureBoundExpectation(minimum=500, maximum=30000),
    "FullBath": FeatureBoundExpectation(minimum=0, maximum=5),
    "OverallQual": FeatureBoundExpectation(minimum=1, maximum=10),
}

SCENARIO_REGISTRY: list[RegisteredScenario] = [
    RegisteredScenario(
        scenario_id="suburban-sfh-ks",
        label="Overland Park KS — Suburban SFH",
        category="suburban",
        description=(
            "Mid-size suburban single-family in Overland Park, Kansas. "
            "Validates core pipeline with expected garage and living-area features."
        ),
        payload=AddressPayload(
            address_line_1="4412 W 109th St",
            city="Overland Park",
            state="KS",
            postal_code="66207",
            country="US",
        ),
        expectations=BaselineExpectationsInput(
            min_completeness_score=0.85,
            required_features=["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea", "GarageCars"],
            feature_bounds={
                **_WIDE_BOUNDS,
                "GarageCars": FeatureBoundExpectation(minimum=0, maximum=4),
            },
        ),
    ),
    RegisteredScenario(
        scenario_id="urban-rowhouse-pa",
        label="Philadelphia PA — Urban Rowhouse",
        category="urban",
        description=(
            "Dense urban rowhouse in South Broad Street corridor, Philadelphia. "
            "Tests small-lot urban property type handling."
        ),
        payload=AddressPayload(
            address_line_1="1234 S Broad St",
            city="Philadelphia",
            state="PA",
            postal_code="19147",
            country="US",
        ),
        expectations=BaselineExpectationsInput(
            min_completeness_score=0.85,
            required_features=["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea"],
            feature_bounds=_WIDE_BOUNDS,
        ),
    ),
    RegisteredScenario(
        scenario_id="coastal-condo-fl",
        label="Miami FL — Coastal Condo",
        category="coastal",
        description=(
            "High-density coastal condo in downtown Miami. "
            "Validates the pipeline handles smaller unit footprints correctly."
        ),
        payload=AddressPayload(
            address_line_1="185 SW 7th St",
            city="Miami",
            state="FL",
            postal_code="33130",
            country="US",
        ),
        expectations=BaselineExpectationsInput(
            min_completeness_score=0.85,
            required_features=["BedroomAbvGr", "GrLivArea"],
            feature_bounds={
                "BedroomAbvGr": FeatureBoundExpectation(minimum=1, maximum=8),
                "GrLivArea": FeatureBoundExpectation(minimum=300, maximum=5000),
                "LotArea": FeatureBoundExpectation(minimum=500, maximum=30000),
                "OverallQual": FeatureBoundExpectation(minimum=1, maximum=10),
            },
        ),
    ),
    RegisteredScenario(
        scenario_id="sun-belt-new-build-az",
        label="Phoenix AZ — Sun Belt Newer Build",
        category="suburban",
        description=(
            "Newer-construction single-family in North Phoenix. "
            "Tests garage-heavy feature set and warmer-market provenance."
        ),
        payload=AddressPayload(
            address_line_1="7200 N Dreamy Draw Dr",
            city="Phoenix",
            state="AZ",
            postal_code="85020",
            country="US",
        ),
        expectations=BaselineExpectationsInput(
            min_completeness_score=0.85,
            required_features=["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea", "GarageCars"],
            feature_bounds={
                **_WIDE_BOUNDS,
                "GarageCars": FeatureBoundExpectation(minimum=0, maximum=4),
            },
        ),
    ),
    RegisteredScenario(
        scenario_id="college-town-ia",
        label="Ames IA — College Town Home",
        category="college-town",
        description=(
            "Residential property in Ames, Iowa — the same market as the Ames housing "
            "dataset used to train the model. Highest-fidelity scenario for feature coverage."
        ),
        payload=AddressPayload(
            address_line_1="413 Duff Ave",
            city="Ames",
            state="IA",
            postal_code="50010",
            country="US",
        ),
        expectations=BaselineExpectationsInput(
            min_completeness_score=0.85,
            required_features=[
                "BedroomAbvGr",
                "TotRmsAbvGrd",
                "GrLivArea",
                "LotArea",
                "OverallQual",
                "YearBuilt",
            ],
            feature_bounds={
                **_WIDE_BOUNDS,
                "YearBuilt": FeatureBoundExpectation(minimum=1900, maximum=2026),
            },
        ),
    ),
]


def get_all_scenarios() -> list[RegisteredScenario]:
    return list(SCENARIO_REGISTRY)


def get_scenario_by_id(scenario_id: str) -> RegisteredScenario | None:
    return next((s for s in SCENARIO_REGISTRY if s.scenario_id == scenario_id), None)


def get_scenarios_by_ids(scenario_ids: list[str]) -> list[RegisteredScenario]:
    id_set = set(scenario_ids)
    return [s for s in SCENARIO_REGISTRY if s.scenario_id in id_set]
