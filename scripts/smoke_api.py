from __future__ import annotations

import json

from fastapi.testclient import TestClient

from house_price_prediction.api.main import create_app
from house_price_prediction.config import load_settings


def _assert_status(name: str, response, expected_status: int) -> None:
    if response.status_code != expected_status:
        raise SystemExit(
            f"{name} failed with status {response.status_code}: {response.text}"
        )


if __name__ == "__main__":
    settings = load_settings()
    with TestClient(create_app(settings)) as client:
        health_response = client.get("/v1/health")
        dashboard_response = client.get("/v1/dashboard/bootstrap?limit=5")
        prediction_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "1600 Pennsylvania Ave NW",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        prediction_detail_response = None
        prediction_trace_response = None
        predictions_list_response = None
        if prediction_response.status_code == 201:
            prediction_id = prediction_response.json()["prediction_id"]
            prediction_detail_response = client.get(f"/v1/predictions/{prediction_id}")
            prediction_trace_response = client.get(f"/v1/predictions/{prediction_id}/trace")
            predictions_list_response = client.get("/v1/predictions?limit=5")
        po_box_response = client.post(
            "/v1/predictions",
            json={
                "address_line_1": "PO Box 123",
                "city": "Washington",
                "state": "DC",
                "postal_code": "20500",
                "country": "US",
            },
        )
        international_response = None
        if settings.geocoding_provider.strip().lower().startswith("free") or settings.property_data_provider.strip().lower().startswith("free"):
            international_response = client.post(
                "/v1/predictions",
                json={
                    "address_line_1": "111 Wellington St",
                    "city": "Ottawa",
                    "state": "ON",
                    "postal_code": "K1A0A9",
                    "country": "CA",
                },
            )

    _assert_status("health", health_response, 200)
    _assert_status("dashboard", dashboard_response, 200)
    _assert_status("prediction", prediction_response, 201)
    if prediction_detail_response is not None:
        _assert_status("prediction_detail", prediction_detail_response, 200)
    if prediction_trace_response is not None:
        _assert_status("prediction_trace", prediction_trace_response, 200)
    if predictions_list_response is not None:
        _assert_status("predictions_list", predictions_list_response, 200)
    _assert_status("po_box_guardrail", po_box_response, 422)
    if international_response is not None:
        _assert_status("international_guardrail", international_response, 422)

    print("Health:")
    print(json.dumps(health_response.json(), indent=2))
    print("Dashboard:")
    print(json.dumps(dashboard_response.json(), indent=2))
    print("Prediction:")
    print(json.dumps(prediction_response.json(), indent=2))
    if prediction_detail_response is not None:
        print("Prediction Detail:")
        print(json.dumps(prediction_detail_response.json(), indent=2))
    if prediction_trace_response is not None:
        print("Prediction Trace:")
        print(json.dumps(prediction_trace_response.json(), indent=2))
    if predictions_list_response is not None:
        print("Predictions List:")
        print(json.dumps(predictions_list_response.json(), indent=2))
    print("PO Box Guardrail:")
    print(json.dumps(po_box_response.json(), indent=2))
    if international_response is not None:
        print("International Guardrail:")
        print(json.dumps(international_response.json(), indent=2))