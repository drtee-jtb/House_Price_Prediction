from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

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