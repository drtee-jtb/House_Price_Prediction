from __future__ import annotations

from uuid import uuid4

from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)


def test_quality_first_policy_biases_completeness_toward_key_features():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    balanced_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="balanced-v1",
    )
    quality_first_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="quality-first-v1",
    )

    balanced_vector = balanced_service.assemble(request_id=request_id, provider_payload=payload)
    quality_first_vector = quality_first_service.assemble(request_id=request_id, provider_payload=payload)

    assert balanced_vector.completeness_score == 0.5
    assert quality_first_vector.completeness_score > balanced_vector.completeness_score
    assert quality_first_vector.feature_policy_name == "quality-first-v1"
    assert quality_first_vector.feature_policy_version == "v1"


def test_unknown_feature_policy_falls_back_to_balanced_weights():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    unknown_policy_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="future-experimental-policy",
    )

    vector = unknown_policy_service.assemble(request_id=request_id, provider_payload=payload)

    assert vector.completeness_score == 0.5
    assert vector.weight_total == 2.0
    assert vector.feature_policy_name == "future-experimental-policy"


def test_state_override_selects_policy_for_context():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="balanced-v1",
        feature_policy_state_overrides={"FL": "quality-first-v1"},
    )

    vector = service.assemble(
        request_id=request_id,
        provider_payload=payload,
        context={"state": "fl"},
    )

    assert vector.feature_policy_name == "quality-first-v1"
    assert vector.completeness_score > 0.5
