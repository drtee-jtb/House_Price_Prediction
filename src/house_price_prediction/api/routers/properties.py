from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from house_price_prediction.api.dependencies import get_prediction_orchestrator, get_settings
from house_price_prediction.api.guardrails import (
    RequestGuardrailError,
    validate_address_payload,
)
from house_price_prediction.application.services.prediction_orchestrator import (
    PredictionOrchestrator,
)
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    NormalizedAddress,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/properties", tags=["properties"])


@router.post("/normalize", response_model=NormalizedAddress)
def normalize_property_address(
    payload: AddressPayload,
    orchestrator: PredictionOrchestrator = Depends(get_prediction_orchestrator),
    settings: Settings = Depends(get_settings),
) -> NormalizedAddress:
    try:
        validate_address_payload(payload, settings)
        return orchestrator.normalize_address(payload)
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc