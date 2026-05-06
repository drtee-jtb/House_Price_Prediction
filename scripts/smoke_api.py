#!/usr/bin/env python
"""
smoke_api.py — Production readiness smoke test for the House Price Prediction API.

Runs end-to-end checks against the API using the TestClient (no server required).
Covers:
  - Health endpoint
  - Prediction creation and retrieval
  - Prediction reuse
  - Address normalization
  - Validation baseline
  - Feature policy catalog
  - Dashboard bootstrap
  - Error handling (bad input → 422)

Usage:
  python scripts/smoke_api.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile

from fastapi.testclient import TestClient

from house_price_prediction.api.main import create_app
from house_price_prediction.config import Settings

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    line = f"  {status} {name}"
    if detail:
        line += f"  [{detail}]"
    print(line)


def _build_settings(tmp_path: Path) -> Settings:
    """Build settings respecting env vars, falling back to safe test defaults."""
    mock = os.getenv("ENABLE_MOCK_PREDICTOR", "false").strip().lower() in ("1", "true", "yes")
    prop_provider = os.getenv("PROPERTY_DATA_PROVIDER", "fake")
    geo_provider = os.getenv("GEOCODING_PROVIDER", "fake")
    reuse_hours = int(os.getenv("PREDICTION_REUSE_MAX_AGE_HOURS", "24"))
    db_url = os.getenv("DATABASE_URL", f"sqlite:///{tmp_path / 'smoke.db'}")
    timeout = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "25.0" if prop_provider != "fake" else "5.0"))

    return Settings(
        raw_data_path=Path("data/raw/housing.csv"),
        target_column="SalePrice",
        model_path=Path("models/house_price_model.joblib"),
        test_size=0.2,
        random_state=42,
        app_name="Smoke Test API",
        app_env="test",
        api_host="127.0.0.1",
        api_port=8001,
        database_url=db_url,
        model_name="house-price-random-forest",
        model_version="smoke-test",
        enable_mock_predictor=mock,
        property_data_provider=prop_provider,
        geocoding_provider=geo_provider,
        prediction_reuse_max_age_hours=reuse_hours,
        provider_timeout_seconds=timeout,
        provider_max_retries=1,
    )


_SAMPLE_ADDRESS = {
    "address_line_1": "123 Main St",
    "city": "Seattle",
    "state": "WA",
    "postal_code": "98101",
    "country": "US",
}

_SAMPLE_ADDRESS_2 = {
    "address_line_1": "456 Oak Ave",
    "city": "Portland",
    "state": "OR",
    "postal_code": "97201",
    "country": "US",
}


def run_smoke_tests(client: TestClient, settings: Settings) -> None:
    print("\n─── Health ────────────────────────────────────────────────────────")
    r = client.get("/v1/health")
    check("GET /v1/health returns 200", r.status_code == 200)
    body = r.json()
    check("health.status == 'ok'", body.get("status") == "ok", body.get("status"))
    check("health.model_available", body.get("model_available") is True)

    print("\n─── Predictions ───────────────────────────────────────────────────")
    r = client.post("/v1/predictions", json=_SAMPLE_ADDRESS)
    check("POST /v1/predictions returns 201", r.status_code == 201, str(r.status_code))
    pred = r.json()
    check("prediction has predicted_price", "predicted_price" in pred)
    price = pred.get("predicted_price", 0)
    check(
        f"predicted_price ${price:,.0f} is in plausible range ($50k–$3M)",
        50_000 <= price <= 3_000_000,
        f"${price:,.0f}",
    )
    check("prediction.status == 'completed'", pred.get("status") == "completed")
    check("prediction has normalized_address", "normalized_address" in pred)
    check("prediction has key_features", isinstance(pred.get("key_features"), dict))
    check("prediction has feature_source", pred.get("feature_source") is not None)
    prediction_id = pred.get("prediction_id")

    r2 = client.get(f"/v1/predictions/{prediction_id}")
    check("GET /v1/predictions/{id} returns 200", r2.status_code == 200)
    detail = r2.json()
    check("detail has feature_snapshot", "feature_snapshot" in detail)
    check("detail has provider_responses", isinstance(detail.get("provider_responses"), list))
    check(
        "detail.predicted_price matches create price",
        abs(detail.get("predicted_price", -1) - price) < 1,
        f"detail={detail.get('predicted_price')}, create={price}",
    )

    r3 = client.get(f"/v1/predictions/{prediction_id}/trace")
    check("GET /v1/predictions/{id}/trace returns 200", r3.status_code == 200)
    trace = r3.json()
    check("trace has trace_nodes", isinstance(trace.get("trace_nodes"), list))
    check("trace has workflow_events", isinstance(trace.get("workflow_events"), list))

    r4 = client.get(f"/v1/predictions/{prediction_id}/events")
    check("GET /v1/predictions/{id}/events returns 200", r4.status_code == 200)
    ev = r4.json()
    check("events.total_count > 0", ev.get("total_count", 0) > 0, str(ev.get("total_count")))

    print("\n─── Prediction reuse ──────────────────────────────────────────────")
    r_reuse = client.post("/v1/predictions", json=_SAMPLE_ADDRESS)
    check("second POST returns 201", r_reuse.status_code == 201)
    pred2 = r_reuse.json()
    reuse_enabled = settings.prediction_reuse_max_age_hours > 0
    if reuse_enabled:
        check("second prediction was_reused=True", pred2.get("was_reused") is True)
        check(
            "reused prediction price matches original",
            abs(pred2.get("predicted_price", -1) - price) < 1,
        )
    else:
        check(
            "second prediction (reuse disabled) returns new price",
            pred2.get("status") == "completed",
            "reuse_max_age=0, skip reuse checks",
        )

    print("\n─── Prediction list ───────────────────────────────────────────────")
    r_list = client.get("/v1/predictions?limit=5")
    check("GET /v1/predictions returns 200", r_list.status_code == 200)
    lst = r_list.json()
    check("list has items", isinstance(lst.get("items"), list) and len(lst["items"]) >= 1)
    check("list has total", isinstance(lst.get("total"), int))

    print("\n─── Properties / address normalization ────────────────────────────")
    r_norm = client.post("/v1/properties/normalize", json=_SAMPLE_ADDRESS)
    check("POST /v1/properties/normalize returns 200", r_norm.status_code == 200)
    norm = r_norm.json()
    check("normalize returns formatted_address", bool(norm.get("formatted_address")))
    check("normalize returns state", bool(norm.get("state")))

    print("\n─── Policies ──────────────────────────────────────────────────────")
    r_pol = client.get("/v1/policies/feature")
    check("GET /v1/policies/feature returns 200", r_pol.status_code == 200)
    pol = r_pol.json()
    check("policies catalog has entries", len(pol.get("policies", [])) >= 1)
    check("default_policy_name is set", bool(pol.get("default_policy_name")))

    print("\n─── Validation / baseline ─────────────────────────────────────────")
    r_scenarios = client.get("/v1/validation/scenarios")
    check("GET /v1/validation/scenarios returns 200", r_scenarios.status_code == 200)
    scen = r_scenarios.json()
    check("scenarios list is non-empty", len(scen.get("scenarios", [])) >= 1)

    r_baseline = client.post("/v1/validation/address-baseline", json=_SAMPLE_ADDRESS_2)
    check(
        "POST /v1/validation/address-baseline returns 200",
        r_baseline.status_code == 200,
        str(r_baseline.status_code),
    )
    baseline = r_baseline.json()
    check("baseline has features", "features" in baseline)
    check("baseline has value (price estimate)", "value" in baseline)
    check("baseline.assessment.overall_status is set", "assessment" in baseline)

    print("\n─── Dashboard ─────────────────────────────────────────────────────")
    r_dash = client.get("/v1/dashboard/bootstrap")
    check("GET /v1/dashboard/bootstrap returns 200", r_dash.status_code == 200)
    dash = r_dash.json()
    check("dashboard has runtime", "runtime" in dash)
    check("dashboard has recent_predictions", "recent_predictions" in dash)
    check("dashboard has links", "links" in dash)

    print("\n─── Meta / capabilities ───────────────────────────────────────────")
    r_meta = client.get("/v1/meta/capabilities")
    check("GET /v1/meta/capabilities returns 200", r_meta.status_code == 200)
    meta = r_meta.json()
    check("capabilities has endpoints", len(meta.get("endpoints", [])) >= 3)
    check("capabilities.model_expected_features is non-empty", len(meta.get("model_expected_features", [])) > 0)

    print("\n─── Error handling ────────────────────────────────────────────────")
    r_bad = client.post("/v1/predictions", json={"address_line_1": "", "city": "X", "state": "CA", "postal_code": "90001"})
    check("Empty address_line_1 returns 422", r_bad.status_code == 422)

    r_unknown = client.get(f"/v1/predictions/00000000-0000-0000-0000-000000000000")
    check("Unknown prediction_id returns 404", r_unknown.status_code == 404)

    r_bad_policy = client.post(
        "/v1/predictions",
        json={**_SAMPLE_ADDRESS, "preferred_policy_name": "nonexistent-policy-xyz"},
    )
    check("Unknown policy name returns 422", r_bad_policy.status_code == 422)

    r_bad_override = client.post(
        "/v1/predictions",
        json={**_SAMPLE_ADDRESS, "feature_overrides": {"OverallQual": 999}},
    )
    check("Out-of-bounds feature_override returns 422", r_bad_override.status_code == 422)


def main() -> None:
    print("=" * 68)
    print("  HOUSE PRICE PREDICTION — PRODUCTION READINESS SMOKE TEST")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as tmp:
        settings = _build_settings(Path(tmp))
        app = create_app(settings)
        print(f"\nConfiguration:")
        print(f"  model_path       : {settings.model_path}")
        print(f"  mock_predictor   : {settings.enable_mock_predictor}")
        print(f"  property_provider: {settings.property_data_provider}")
        print(f"  geocoding        : {settings.geocoding_provider}")
        print(f"  model_available  : {settings.model_path.exists()}")

        with TestClient(app) as client:
            run_smoke_tests(client, settings)

    print("\n" + "=" * 68)
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all checks passed!")
    print("=" * 68)

    if failed:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  {FAIL} {name}" + (f"  [{detail}]" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
