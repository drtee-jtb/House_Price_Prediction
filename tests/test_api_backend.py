from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from time import sleep
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from house_price_prediction.api.main import create_app
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.census_property_data_client import (
    CensusPropertyDataClient,
)
from house_price_prediction.infrastructure.db.repositories import PredictionRepository
from house_price_prediction.infrastructure.db.session import init_database
from house_price_prediction.infrastructure.providers.fallback_geocoding_provider import (
    FallbackGeocodingProvider,
)
from house_price_prediction.infrastructure.providers.fallback_property_data_provider import (
    FallbackPropertyDataProvider,
)
from house_price_prediction.infrastructure.providers.fake_geocoding_client import FakeGeocodingClient
from house_price_prediction.infrastructure.providers.fake_property_data_client import (
    FakePropertyDataClient,
)
from house_price_prediction.infrastructure.providers.heuristic_property_data_client import (
    HeuristicPropertyDataClient,
)
from house_price_prediction.infrastructure.providers.resilient import (
    ProviderExecutionError,
    ResilientPropertyDataProvider,
)
from house_price_prediction.model import load_model_artifact, save_model_artifact, train_and_save_model


class FailingPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        raise RuntimeError("boom")


class SlowPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        sleep(0.05)
        return ProviderResponseContract(
            provider_name="slow",
            status="success",
            payload={"LotArea": 1000},
            fetched_at=datetime.now(UTC),
        )


class FailingGeocoder:
    def normalize(self, address_payload):
        raise RuntimeError("no match")


class AlwaysFailPropertyProvider:
    def fetch_property_features(self, normalized_address) -> ProviderResponseContract:
        raise RuntimeError("upstream unavailable")


def build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_path=Path("data/raw/housing.csv"),
        target_column="SalePrice",
        model_path=tmp_path / "model.joblib",
        test_size=0.2,
        random_state=42,
        app_name="House Price Prediction API Test",
        app_env="test",
        api_host="127.0.0.1",
        api_port=8001,
        database_url=f"sqlite:///{tmp_path / 'backend_test.db'}",
        model_name="house-price-random-forest",
        model_version="test-version",
        enable_mock_predictor=True,
        property_data_provider="fake",
        geocoding_provider="fake",
        prediction_reuse_max_age_hours=24,
        provider_timeout_seconds=3.0,
        provider_max_retries=2,
    )


def test_prediction_endpoint_persists_and_fetches_result(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
                "requested_by": "team@example.com",
            },
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["predicted_price"] > 0
        assert payload["model_version"] == "test-version"

        fetch_response = client.get(f"/v1/predictions/{payload['prediction_id']}")
        assert fetch_response.status_code == 200
        fetch_payload = fetch_response.json()
        assert fetch_payload["request_id"] == payload["request_id"]
        assert fetch_payload["predicted_price"] == payload["predicted_price"]
        assert fetch_payload["normalized_address"]["city"] == "MIAMI"
        assert fetch_payload["normalized_address"]["geocoding_source"] == "fake"
        assert fetch_payload["feature_snapshot"]["feature_count"] >= 1
        assert fetch_payload["provider_responses"][1]["feature_source"] == "fake"
        assert fetch_payload["provider_responses"][1]["feature_provenance"]["strategy"] == "deterministic_fake"
        assert fetch_payload["source_prediction_id"] is None


def test_health_endpoint_reports_runtime_state(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["mock_predictor_enabled"] is True
        assert payload["model_available"] is True
        assert payload["property_data_provider"] == "fake"
        assert payload["geocoding_provider"] == "fake"
        assert payload["provider_timeout_seconds"] == 3.0
        assert payload["provider_max_retries"] == 2
        assert payload["feature_policy_name"] == "balanced-v1"
        assert payload["feature_policy_version"] == "v1"
        assert payload["feature_policy_state_override_count"] == 0


def test_predictions_list_endpoint_returns_recent_predictions(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        for address_line_1 in ["123 Main St", "456 Oak Ave"]:
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Miami",
                    "state": "FL",
                    "postal_code": "33101",
                    "country": "US",
                },
            )
            assert response.status_code == 201

        list_response = client.get("/v1/predictions?limit=2")

    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload["items"]) == 2
    assert payload["items"][0]["feature_source"] == "fake"
    assert payload["items"][0]["normalized_address"]["country"] == "US"


def test_dashboard_bootstrap_returns_runtime_policy_and_recent_predictions(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )
        assert create_response.status_code == 201

        response = client.get("/v1/dashboard/bootstrap?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["runtime"]["app_name"] == settings.app_name
    assert payload["runtime"]["model_version"] == settings.model_version
    assert payload["provider_policy"]["geocoding_provider"] == settings.geocoding_provider
    assert payload["provider_policy"]["property_data_provider"] == settings.property_data_provider
    assert payload["provider_policy"]["feature_policy_name"] == settings.feature_policy_name
    assert payload["provider_policy"]["feature_policy_version"] == settings.feature_policy_version
    assert payload["provider_policy"]["feature_policy_state_overrides"] == settings.feature_policy_state_overrides
    assert payload["links"]["prediction_detail"] == "/v1/predictions/{prediction_id}"
    assert len(payload["recent_predictions"]) == 1
    assert payload["recent_predictions"][0]["normalized_address"]["city"] == "MIAMI"
    assert payload["event_summary"]["prediction_id"] == payload["recent_predictions"][0]["prediction_id"]
    assert payload["event_summary"]["total_events"] >= 1
    assert payload["event_summary"]["latest_event_name"] is not None
    assert isinstance(payload["event_summary"]["recent_event_names"], list)


def test_reused_prediction_detail_surfaces_source_lineage(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        second_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )

        assert first_response.status_code == 201
        assert second_response.status_code == 201
        second_payload = second_response.json()
        assert second_payload["was_reused"] is True
        assert second_payload["source_prediction_id"] == first_response.json()["prediction_id"]

        detail_response = client.get(f"/v1/predictions/{second_payload['prediction_id']}")

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["was_reused"] is True
    assert detail_payload["source_prediction_id"] == first_response.json()["prediction_id"]
    assert len(detail_payload["provider_responses"]) == 2
    assert detail_payload["provider_responses"][1]["feature_source"] == "fake"


def test_prediction_trace_endpoint_returns_lineage_nodes(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        second_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )

        trace_response = client.get(
            f"/v1/predictions/{second_response.json()['prediction_id']}/trace"
        )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert trace_payload["was_reused"] is True
    assert trace_payload["root_prediction_id"] == first_response.json()["prediction_id"]
    assert len(trace_payload["trace_nodes"]) == 2
    assert trace_payload["trace_nodes"][0]["prediction_id"] == second_response.json()["prediction_id"]
    assert trace_payload["trace_nodes"][1]["prediction_id"] == first_response.json()["prediction_id"]
    assert trace_payload["provider_responses"][1]["provider_name"] == "fake_property_data"


def test_prediction_persists_provider_response(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "456 Oak Ave",
                "city": "Orlando",
                "state": "FL",
                "postal_code": "32801",
                "country": "US",
            },
        )

        assert response.status_code == 201
        payload = response.json()

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        provider_responses = repository.get_provider_responses(payload["request_id"])

    assert len(provider_responses) == 2
    assert provider_responses[0].provider_name == "fake_geocoding"
    assert provider_responses[0].status == "success"
    assert "normalized" in provider_responses[0].payload
    assert provider_responses[1].provider_name == "fake_property_data"
    assert provider_responses[1].status == "success"
    assert "LotArea" in provider_responses[1].payload
    assert provider_responses[1].payload["feature_provenance"]["strategy"] == "deterministic_fake"


def test_model_artifact_preserves_feature_metadata(tmp_path: Path):
    model_path = tmp_path / "artifact.joblib"

    save_model_artifact(
        model={"kind": "dummy-model"},
        model_path=model_path,
        feature_columns=["LotArea", "OverallQual", "Neighborhood"],
        target_column="SalePrice",
        model_name="house-price-random-forest",
        model_version="2026.04",
    )

    artifact = load_model_artifact(model_path)

    assert artifact.model == {"kind": "dummy-model"}
    assert artifact.metadata.feature_columns == ("LotArea", "OverallQual", "Neighborhood")
    assert artifact.metadata.target_column == "SalePrice"
    assert artifact.metadata.model_version == "2026.04"


def test_repeated_requests_reuse_canonical_normalized_address_and_register_model(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    responses = []
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": " 789 Pine Road ",
                    "city": "Tampa",
                    "state": "fl",
                    "postal_code": "33602",
                    "country": "us",
                },
            )
            assert response.status_code == 201
            responses.append(response.json())

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        assert repository.count_normalized_addresses() == 1

        registry_entry = repository.get_model_registry_entry(
            model_name=settings.model_name,
            model_version=settings.model_version,
        )

    assert registry_entry is not None
    assert registry_entry.is_active is True
    assert "LotArea" in registry_entry.feature_columns
    assert responses[0]["was_reused"] is False
    assert responses[1]["was_reused"] is True
    assert responses[0]["predicted_price"] == responses[1]["predicted_price"]


def test_reuse_can_be_disabled_with_zero_hour_freshness_window(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    settings = Settings(
        raw_data_path=settings.raw_data_path,
        target_column=settings.target_column,
        model_path=settings.model_path,
        test_size=settings.test_size,
        random_state=settings.random_state,
        app_name=settings.app_name,
        app_env=settings.app_env,
        api_host=settings.api_host,
        api_port=settings.api_port,
        database_url=settings.database_url,
        model_name=settings.model_name,
        model_version=settings.model_version,
        enable_mock_predictor=settings.enable_mock_predictor,
        property_data_provider=settings.property_data_provider,
        geocoding_provider=settings.geocoding_provider,
        prediction_reuse_max_age_hours=0,
        provider_timeout_seconds=settings.provider_timeout_seconds,
        provider_max_retries=settings.provider_max_retries,
    )

    responses = []
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": "22 Sunset Blvd",
                    "city": "Miami",
                    "state": "FL",
                    "postal_code": "33101",
                    "country": "US",
                },
            )
            assert response.status_code == 201
            responses.append(response.json())

    assert responses[0]["was_reused"] is False
    assert responses[1]["was_reused"] is False


def test_reuse_does_not_cross_feature_policy_boundaries(tmp_path: Path):
    base_settings = build_test_settings(tmp_path)
    first_policy_settings = replace(
        base_settings,
        feature_policy_name="balanced-v1",
    )
    second_policy_settings = replace(
        base_settings,
        feature_policy_name="quality-first-v1",
    )

    with TestClient(create_app(first_policy_settings)) as first_client:
        first_response = first_client.post(
            "/v1/predictions",
            json={
                "address_line_1": "55 Policy Way",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    with TestClient(create_app(second_policy_settings)) as second_client:
        second_response = second_client.post(
            "/v1/predictions",
            json={
                "address_line_1": "55 Policy Way",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["was_reused"] is False
    assert second_response.json()["was_reused"] is False
    assert first_response.json()["selected_feature_policy_name"] == "balanced-v1"
    assert second_response.json()["selected_feature_policy_name"] == "quality-first-v1"


def test_normalize_route_uses_geocoding_provider(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": " 123 main st ",
                "city": "miami",
                "state": "fl",
                "postal_code": "33101",
                "country": "us",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["address_line_1"] == "123 MAIN ST"
    assert payload["city"] == "MIAMI"
    assert payload["formatted_address"] == "123 MAIN ST, MIAMI, FL 33101, US"


def test_prediction_maps_provider_failure_to_bad_gateway(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    failing_provider = ResilientPropertyDataProvider(
        provider_name="failing",
        delegate=FailingPropertyProvider(),
        timeout_seconds=0.1,
        max_retries=1,
    )

    with TestClient(create_app(settings, property_data_provider=failing_provider)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 502
    assert "Provider 'failing' failed" in response.json()["detail"]


def test_prediction_persists_failed_provider_snapshot_and_failed_request_status(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    failing_provider = ResilientPropertyDataProvider(
        provider_name="failing",
        delegate=FailingPropertyProvider(),
        timeout_seconds=0.1,
        max_retries=1,
    )
    correlation_id = "11111111-1111-4111-8111-111111111111"

    with TestClient(create_app(settings, property_data_provider=failing_provider)) as client:
        response = client.post(
            "/v1/predictions",
            headers={"x-correlation-id": correlation_id},
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 502

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        repository = PredictionRepository(session)
        request_id = repository.get_request_id_by_correlation_id(correlation_id)

        assert request_id is not None
        assert repository.get_request_status(request_id) == "failed"

        provider_responses = repository.get_provider_responses(request_id)

    assert len(provider_responses) == 2
    assert provider_responses[0].provider_name == "fake_geocoding"
    assert provider_responses[0].status == "success"
    assert provider_responses[1].provider_name == "failing"
    assert provider_responses[1].status == "failed"
    assert provider_responses[1].payload["stage"] == "provider_execution"


def test_normalize_route_maps_provider_failure_to_bad_gateway(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(
        create_app(
            settings,
            geocoding_provider=type(
                "AlwaysFailGeocoder",
                (),
                {
                    "normalize": lambda self, payload: (_ for _ in ()).throw(
                        ProviderExecutionError("failing-geocoder", "Provider 'failing-geocoder' failed: no match")
                    )
                },
            )(),
        )
    ) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": "123 main st",
                "city": "miami",
                "state": "fl",
                "postal_code": "33101",
                "country": "us",
            },
        )

    assert response.status_code == 502
    assert "failing-geocoder" in response.json()["detail"]


def test_resilient_provider_times_out(tmp_path: Path):
    settings = build_test_settings(tmp_path)
    slow_provider = ResilientPropertyDataProvider(
        provider_name="slow",
        delegate=SlowPropertyProvider(),
        timeout_seconds=0.01,
        max_retries=0,
    )

    with TestClient(create_app(settings, property_data_provider=slow_provider)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "100 Ocean Dr",
                "city": "Miami Beach",
                "state": "FL",
                "postal_code": "33139",
                "country": "US",
            },
        )

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]


def test_fallback_geocoder_uses_next_provider():
    provider = FallbackGeocodingProvider(
        providers=(FailingGeocoder(), FakeGeocodingClient())
    )

    result = provider.normalize(
        address_payload=type(
            "Payload",
            (),
            {
                "address_line_1": "1 Main St",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )()
    )

    assert isinstance(result, GeocodingResultContract)
    assert result.normalized_address.formatted_address == "1 MAIN ST, MIAMI, FL 33101, US"
    assert result.normalized_address.geocoding_source == "fake"


def test_census_property_provider_falls_back_without_coordinates():
    provider = CensusPropertyDataClient(fallback_provider=FailingPropertyProvider())
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    try:
        provider.fetch_property_features(normalized_address)
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True


def test_census_property_provider_uses_fake_backfill_without_coordinates():
    provider = CensusPropertyDataClient(fallback_provider=FakePropertyDataClient())
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.status == "success"
    assert response.provider_name.endswith("fallback")
    assert "LotArea" in response.payload
    assert response.payload["feature_provenance"]["used_backfill"] is True
    assert response.payload["feature_provenance"]["used_census"] is False


def test_property_fallback_uses_next_provider():
    provider = FallbackPropertyDataProvider(
        providers=(AlwaysFailPropertyProvider(), HeuristicPropertyDataClient(), FakePropertyDataClient())
    )
    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="MIAMI",
        state="FL",
        postal_code="33101",
        country="US",
        formatted_address="1 MAIN ST, MIAMI, FL 33101, US",
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.status == "success"
    assert response.provider_name == "heuristic_property_data"
    assert response.payload["LotArea"] >= 5000
    assert response.payload["feature_source"] == "heuristic"
    assert response.payload["feature_provenance"]["strategy"] == "heuristic"


def test_census_property_provider_annotates_census_and_backfill_provenance():
    provider = CensusPropertyDataClient(fallback_provider=HeuristicPropertyDataClient())

    provider._lookup_census_tract = lambda latitude, longitude: {
        "state": "11",
        "county": "001",
        "tract": "980000",
        "name": "Downtown, Sample",
    }
    provider._fetch_census_context = lambda geography: {
        "B25077_001E": "650000",
        "B25035_001E": "1998",
        "B25018_001E": "7.1",
    }

    normalized_address = NormalizedAddress(
        address_line_1="1 MAIN ST",
        city="WASHINGTON",
        state="DC",
        postal_code="20500",
        country="US",
        formatted_address="1 MAIN ST, WASHINGTON, DC 20500, US",
        latitude=38.8976,
        longitude=-77.0365,
    )

    response = provider.fetch_property_features(normalized_address)

    assert response.provider_name == "census_context_with_backfill"
    assert response.payload["feature_source"] == "census_context"
    assert response.payload["feature_provenance"]["used_census"] is True
    assert response.payload["feature_provenance"]["used_backfill"] is True
    assert response.payload["feature_provenance"]["backfill_provider"] == "heuristic_property_data"


def test_prediction_guardrail_rejects_po_box(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "P.O. Box 123",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 422
    assert "PO boxes" in response.json()["detail"]


def test_prediction_guardrail_rejects_non_us_for_free_providers(tmp_path: Path):
    base_settings = build_test_settings(tmp_path)
    settings = Settings(
        raw_data_path=base_settings.raw_data_path,
        target_column=base_settings.target_column,
        model_path=base_settings.model_path,
        test_size=base_settings.test_size,
        random_state=base_settings.random_state,
        app_name=base_settings.app_name,
        app_env=base_settings.app_env,
        api_host=base_settings.api_host,
        api_port=base_settings.api_port,
        database_url=base_settings.database_url,
        model_name=base_settings.model_name,
        model_version=base_settings.model_version,
        enable_mock_predictor=base_settings.enable_mock_predictor,
        property_data_provider="free-fallback",
        geocoding_provider="free-fallback",
        prediction_reuse_max_age_hours=base_settings.prediction_reuse_max_age_hours,
        provider_timeout_seconds=base_settings.provider_timeout_seconds,
        provider_max_retries=base_settings.provider_max_retries,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "111 Wellington St",
                "city": "Ottawa",
                "state": "ON",
                "postal_code": "K1A0A9",
                "country": "CA",
            },
        )

    assert response.status_code == 422
    assert "only US addresses" in response.json()["detail"]


def test_normalize_guardrail_rejects_po_box(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/properties/normalize",
            json={
                "address_line_1": "PO Box 88",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 422
    assert "PO boxes" in response.json()["detail"]


def test_non_mock_prediction_path_with_trained_artifact(tmp_path: Path):
    data_path = tmp_path / "housing.csv"
    pd = __import__("pandas")

    training_data = pd.DataFrame(
        {
            "LotArea": [8500, 9600, 11250, 9550, 14260, 14115],
            "OverallQual": [7, 6, 7, 7, 8, 5],
            "OverallCond": [5, 8, 5, 5, 5, 5],
            "YearBuilt": [2003, 1976, 2001, 1915, 2000, 1993],
            "YearRemodAdd": [2003, 1976, 2002, 1970, 2000, 1995],
            "GrLivArea": [1710, 1262, 1786, 1717, 2198, 1362],
            "FullBath": [2, 2, 2, 1, 2, 1],
            "HalfBath": [1, 0, 1, 0, 1, 1],
            "BedroomAbvGr": [3, 3, 3, 3, 4, 3],
            "TotRmsAbvGrd": [8, 6, 6, 7, 9, 7],
            "Fireplaces": [0, 1, 1, 1, 1, 0],
            "GarageCars": [2, 2, 2, 3, 3, 2],
            "GarageArea": [548, 460, 608, 642, 836, 480],
            "Neighborhood": ["CollgCr", "Veenker", "CollgCr", "Crawfor", "NoRidge", "Mitchel"],
            "HouseStyle": ["2Story", "1Story", "2Story", "2Story", "2Story", "1Story"],
            "SalePrice": [208500, 181500, 223500, 140000, 250000, 143000],
        }
    )
    training_data.to_csv(data_path, index=False)

    settings = Settings(
        raw_data_path=data_path,
        target_column="SalePrice",
        model_path=tmp_path / "real_model.joblib",
        test_size=0.2,
        random_state=42,
        app_name="House Price Prediction API Test",
        app_env="test",
        api_host="127.0.0.1",
        api_port=8001,
        database_url=f"sqlite:///{tmp_path / 'non_mock_test.db'}",
        model_name="house-price-random-forest",
        model_version="real-test-version",
        enable_mock_predictor=False,
        property_data_provider="fake",
        geocoding_provider="fake",
        prediction_reuse_max_age_hours=24,
        provider_timeout_seconds=3.0,
        provider_max_retries=2,
    )

    train_and_save_model(settings)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "123 Main St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
                "country": "US",
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["predicted_price"] > 0
    assert payload["model_version"] == "real-test-version"


def test_prediction_persists_brain_workflow_events(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "300 Biscayne Blvd",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 201
    request_id = response.json()["request_id"]

    session_factory = init_database(settings.database_url)
    with session_factory() as session:
        rows = session.execute(
            text(
                """
                SELECT event_name
                FROM workflow_events
                WHERE request_id = :request_id
                ORDER BY occurred_at ASC
                """
            ),
            {"request_id": request_id},
        ).fetchall()

    event_names = [row[0] for row in rows]
    assert "prediction_received" in event_names
    assert "address_normalized" in event_names
    assert "reuse_candidate_evaluated" in event_names
    assert "property_enrichment_completed" in event_names
    assert "feature_vector_created" in event_names
    assert "prediction_completed" in event_names


def test_prediction_workflow_events_endpoint_returns_event_stream(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        events_response = client.get(f"/v1/predictions/{prediction_id}/events")

    assert events_response.status_code == 200
    payload = events_response.json()
    assert payload["prediction_id"] == prediction_id
    assert payload["total_count"] >= 4
    assert payload["limit"] == 100
    assert payload["offset"] == 0
    assert payload["event_name"] is None
    assert len(payload["events"]) >= 4
    event_names = [item["event_name"] for item in payload["events"]]
    assert "prediction_received" in event_names
    assert "prediction_completed" in event_names


def test_prediction_workflow_events_endpoint_supports_filter_and_pagination(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "900 Bayfront Pkwy",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33132",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        filtered_response = client.get(
            f"/v1/predictions/{prediction_id}/events",
            params={"event_name": "prediction_completed", "limit": 1, "offset": 0},
        )

    assert filtered_response.status_code == 200
    payload = filtered_response.json()
    assert payload["event_name"] == "prediction_completed"
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["sort"] == "asc"
    assert payload["total_count"] >= 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["event_name"] == "prediction_completed"


def test_prediction_workflow_events_endpoint_supports_desc_sort(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Harbor Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33132",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        events_response = client.get(
            f"/v1/predictions/{prediction_id}/events",
            params={"sort": "desc", "limit": 1},
        )

    assert events_response.status_code == 200
    payload = events_response.json()
    assert payload["sort"] == "desc"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["event_name"] == "prediction_completed"


def test_prediction_workflow_events_endpoint_returns_404_for_unknown_prediction(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/v1/predictions/{uuid4()}/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found."


def test_prediction_trace_includes_workflow_events(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        create_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "101 Ocean Dr",
                "city": "Miami Beach",
                "state": "FL",
                "postal_code": "33139",
                "country": "US",
            },
        )
        assert create_response.status_code == 201
        prediction_id = create_response.json()["prediction_id"]

        trace_response = client.get(f"/v1/predictions/{prediction_id}/trace")

    assert trace_response.status_code == 200
    payload = trace_response.json()
    assert "workflow_events" in payload
    assert len(payload["workflow_events"]) >= 1
    assert payload["workflow_events"][0]["event_name"] == "prediction_received"


def test_feature_policy_catalog_endpoint_returns_runtime_defaults_and_definitions(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/policies/feature")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_policy_name"] == settings.feature_policy_name
    assert payload["default_policy_version"] == settings.feature_policy_version
    assert payload["state_overrides"] == {}

    policy_names = [item["name"] for item in payload["policies"]]
    assert "balanced-v1" in policy_names
    assert "quality-first-v1" in policy_names
    assert "land-first-v1" in policy_names


def test_feature_policy_simulation_endpoint_returns_requested_policy_results(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1200 Brickell Bay Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "policy_names": ["balanced-v1", "quality-first-v1"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_address"]["city"] == "MIAMI"
    assert payload["provider_name"] == "fake_property_data"
    assert len(payload["simulations"]) == 2

    returned_policy_names = [item["policy_name"] for item in payload["simulations"]]
    assert returned_policy_names == ["balanced-v1", "quality-first-v1"]
    for item in payload["simulations"]:
        assert item["predicted_price"] > 0
        assert 0 <= item["completeness_score"] <= 1
        assert item["weight_total"] is not None


def test_feature_policy_simulation_endpoint_rejects_unknown_policy_name(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/policies/feature/simulate",
            json={
                "address_line_1": "1200 Brickell Bay Dr",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "policy_names": ["unknown-v1"],
            },
        )

    assert response.status_code == 422
    assert "Unsupported feature policy name" in response.json()["detail"]


def test_validation_scenarios_endpoint_returns_baseline_catalog(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/validation/scenarios")

    assert response.status_code == 200
    payload = response.json()
    assert "scenarios" in payload
    assert isinstance(payload["scenarios"], list)


def test_address_baseline_endpoint_returns_location_feature_and_value_contract(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2026-04-16"
    assert payload["location"]["normalized_address"]["city"] == "MIAMI"
    assert payload["location"]["geocoding_provider"] == "fake_geocoding"
    assert payload["features"]["pulled_feature_count"] >= 1
    assert payload["features"]["missing_feature_count"] >= 0
    assert payload["features"]["pulled_feature_count"] + payload["features"]["missing_feature_count"] == len(payload["features"]["expected_features"])
    assert "BedroomAbvGr" in payload["features"]["key_feature_values"]
    assert "TotRmsAbvGrd" in payload["features"]["key_feature_values"]
    assert "GrLivArea" in payload["features"]["key_feature_values"]
    assert payload["value"]["predicted_price"] > 0
    assert payload["value"]["property_provider"] == "fake_property_data"
    assert payload["assessment"]["overall_status"] == "not_evaluated"


def test_address_baseline_with_scenario_returns_assessment_checks(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "expectations": {
                    "min_completeness_score": 0.98,
                    "required_features": ["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea"],
                    "feature_bounds": {
                        "BedroomAbvGr": {"minimum": 1, "maximum": 8},
                        "TotRmsAbvGrd": {"minimum": 2, "maximum": 14},
                        "GrLivArea": {"minimum": 500, "maximum": 6000},
                        "LotArea": {"minimum": 1000, "maximum": 50000},
                    },
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["assessment"]["checks"]) >= 1
    check_names = [item["check_name"] for item in payload["assessment"]["checks"]]
    assert "completeness_threshold" in check_names
    assert any(name.startswith("required_feature:") for name in check_names)
    assert any(name.startswith("feature_bounds:") for name in check_names)


def test_full_audit_endpoint_returns_aggregated_report(tmp_path: Path):
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "500 Brickell Ave",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33131",
                "country": "US",
                "requested_by": "audit@example.com",
                "expectations": {
                    "min_completeness_score": 0.98,
                    "required_features": ["BedroomAbvGr", "TotRmsAbvGrd", "GrLivArea", "LotArea"],
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2026-04-16"
    assert payload["prediction"]["status"] == "completed"
    assert payload["baseline"]["assessment"]["overall_status"] in {"passed", "failed"}
    assert isinstance(payload["issues"], list)


def test_run_scenario_batch_all_registered_scenarios_pass(tmp_path: Path):
    """
    POST /v1/validation/run-scenario-batch with no filter should execute all
    registered scenarios and return a well-formed batched result.  In test
    mode (fake providers) every scenario is expected to pass because the
    expectations are calibrated to the fake provider output range.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/validation/run-scenario-batch", json={})

    assert response.status_code == 200
    payload = response.json()

    assert payload["contract_version"] == "2026-04-16"
    assert isinstance(payload["total"], int) and payload["total"] > 0
    assert payload["errors"] == 0, f"Unexpected errors in batch: {payload.get('results')}"
    assert payload["failed"] == 0, (
        f"Unexpected failures in batch: "
        f"{[r for r in payload['results'] if r['pipeline_status'] == 'fail']}"
    )
    assert payload["passed"] == payload["total"]

    for result in payload["results"]:
        assert result["pipeline_status"] == "pass"
        assert result["completeness_score"] is not None
        assert result["predicted_price"] is not None
        assert result["issues"] == []
        assert isinstance(result["key_feature_values"], dict)


def test_run_scenario_batch_single_scenario_by_id(tmp_path: Path):
    """Filtering by scenario_id should return only the specified scenario."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/run-scenario-batch",
            json={"scenario_ids": ["college-town-ia"]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["results"][0]["scenario_id"] == "college-town-ia"
    assert payload["results"][0]["pipeline_status"] == "pass"


# ---------------------------------------------------------------------------
# Regression tests for bugs fixed in the audit pass
# ---------------------------------------------------------------------------

def test_baseline_with_null_completeness_score_does_not_raise(tmp_path: Path):
    """Regression for Bug 1: passing expectations with min_completeness_score=None
    must not raise TypeError when the completeness check is skipped.
    The baseline should still evaluate required_features checks.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "100 E Ohio St",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60611",
                "country": "US",
                "expectations": {
                    # No completeness threshold — only required-features check
                    "min_completeness_score": None,
                    "required_features": ["BedroomAbvGr", "GrLivArea"],
                    "feature_bounds": {},
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    # No completeness_threshold check should exist in checks
    check_names = [c["check_name"] for c in payload["assessment"]["checks"]]
    assert "completeness_threshold" not in check_names
    # Required feature checks must be present
    assert any(n.startswith("required_feature:") for n in check_names)
    # Overall status must be a valid value, not an error
    assert payload["assessment"]["overall_status"] in {"passed", "failed"}


def test_full_audit_passing_baseline_with_missing_features_has_no_issues(tmp_path: Path):
    """Regression for Bug 2: when expectations are provided and the baseline
    assessment passes (threshold met), run_full_audit must NOT append an extra
    'Missing model features' issue entry just because some features were absent.
    The batch runner would falsely mark those scenarios as 'fail'.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        # Use a low completeness threshold that will definitely pass,
        # ensuring the baseline assessment is "passed" even if some features miss.
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "300 W Grand Ave",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60654",
                "country": "US",
                "expectations": {
                    "min_completeness_score": 0.01,  # near-zero: will definitely pass
                    "required_features": [],
                    "feature_bounds": {},
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["baseline"]["assessment"]["overall_status"] == "passed"
    # A passing baseline must not generate a spurious "Missing model features" issue
    for issue in payload["issues"]:
        assert "Missing model features" not in issue, (
            f"Spurious missing-feature issue found despite passing baseline: {issue}"
        )


def test_full_audit_not_evaluated_baseline_surfaces_missing_feature_count(tmp_path: Path):
    """When no expectations are given (not_evaluated), missing features should
    be surfaced in issues so the caller knows the data quality state.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        # Send no expectations at all
        response = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "413 Duff Ave",
                "city": "Ames",
                "state": "IA",
                "postal_code": "50010",
                "country": "US",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["baseline"]["assessment"]["overall_status"] == "not_evaluated"
    # Fake provider returns all features → missing count is 0 → no issue appended.
    # This asserts the contract: if 0 missing, no issue; if > 0, issue IS present.
    missing_count = payload["baseline"]["features"]["missing_feature_count"]
    has_missing_issue = any("Missing model features" in i for i in payload["issues"])
    if missing_count == 0:
        assert not has_missing_issue
    else:
        assert has_missing_issue


def test_baseline_contract_version_is_consistent(tmp_path: Path):
    """contract_version in baseline and full-audit responses must match."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        baseline_resp = client.post(
            "/v1/validation/address-baseline",
            json={
                "address_line_1": "1 Infinite Loop",
                "city": "Cupertino",
                "state": "CA",
                "postal_code": "95014",
                "country": "US",
            },
        )
        audit_resp = client.post(
            "/v1/validation/full-audit",
            json={
                "address_line_1": "1 Infinite Loop",
                "city": "Cupertino",
                "state": "CA",
                "postal_code": "95014",
                "country": "US",
            },
        )

    assert baseline_resp.status_code == 200
    assert audit_resp.status_code == 200
    assert baseline_resp.json()["contract_version"] == audit_resp.json()["contract_version"]
    assert baseline_resp.json()["contract_version"] == "2026-04-16"


def test_init_database_is_migration_first_by_default(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'migration_first.db'}"
    session_factory = init_database(database_url)

    with session_factory() as session:
        table_rows = session.execute(
            text(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        ).fetchall()

    table_names = {row[0] for row in table_rows}
    assert "prediction_requests" not in table_names
    assert "workflow_events" not in table_names


def test_predictions_list_response_includes_pagination_metadata(tmp_path: Path):
    """PredictionListResponse must carry total, limit, and offset fields."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        for address_line_1 in ["1 Alpha Rd", "2 Beta Rd", "3 Gamma Rd"]:
            r = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Denver",
                    "state": "CO",
                    "postal_code": "80201",
                    "country": "US",
                },
            )
            assert r.status_code == 201

        # First page: limit=2, offset=0
        first_page = client.get("/v1/predictions?limit=2&offset=0")
    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["total"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert len(payload["items"]) == 2


def test_predictions_list_offset_returns_correct_page(tmp_path: Path):
    """offset parameter must skip the right number of rows."""
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        ids_in_order: list[str] = []
        for address_line_1 in ["10 First St", "20 Second St", "30 Third St"]:
            r = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": address_line_1,
                    "city": "Austin",
                    "state": "TX",
                    "postal_code": "78701",
                    "country": "US",
                },
            )
            assert r.status_code == 201
            ids_in_order.append(r.json()["prediction_id"])

        all_items = client.get("/v1/predictions?limit=50&offset=0").json()["items"]
        second_page = client.get("/v1/predictions?limit=2&offset=2").json()

    # Results are newest-first, so third (last submitted) is at index 0 in all_items.
    assert second_page["offset"] == 2
    assert len(second_page["items"]) == 1
    assert second_page["items"][0]["prediction_id"] == all_items[2]["prediction_id"]


def test_fresh_prediction_has_null_confidence_score(tmp_path: Path):
    """confidence_score must be None for fresh (non-reused) predictions.

    Completeness score is a feature-coverage metric, not model confidence.
    Storing them as the same field misleads API consumers.
    """
    settings = build_test_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        r = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "77 Oak Lane",
                "city": "Portland",
                "state": "OR",
                "postal_code": "97201",
                "country": "US",
            },
        )
        assert r.status_code == 201
        prediction_id = r.json()["prediction_id"]
        assert r.json()["confidence_score"] is None

        detail = client.get(f"/v1/predictions/{prediction_id}")
    assert detail.status_code == 200
    assert detail.json()["confidence_score"] is None
    # completeness_score still reaches the caller via the feature_snapshot
    assert detail.json()["feature_snapshot"]["completeness_score"] > 0.0