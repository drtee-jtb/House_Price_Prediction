from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from house_price_prediction.api.dependencies import get_brain, get_settings
from house_price_prediction.api.guardrails import RequestGuardrailError, validate_address_payload
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    FeaturePolicyCatalogResponse,
    FeaturePolicySimulationRequest,
    FeaturePolicySimulationResponse,
)
from house_price_prediction.infrastructure.model_runtime.predictor import (
    ModelInferenceError,
    ModelNotReadyError,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/policies", tags=["policies"])


@router.get("/feature", response_model=FeaturePolicyCatalogResponse)
def get_feature_policy_catalog(
    brain: Brain = Depends(get_brain),
) -> FeaturePolicyCatalogResponse:
    return brain.get_feature_policy_catalog()


@router.post(
    "/feature/simulate",
    response_model=FeaturePolicySimulationResponse,
    status_code=status.HTTP_200_OK,
)
def simulate_feature_policies(
    payload: FeaturePolicySimulationRequest,
    brain: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> FeaturePolicySimulationResponse:
    try:
        validate_address_payload(payload, settings)
        return brain.simulate_feature_policies(payload)
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ModelInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
