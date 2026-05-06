from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from house_price_prediction.api.dependencies import get_brain, get_settings
from house_price_prediction.api.guardrails import RequestGuardrailError, validate_address_payload
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    NormalizedAddress,
)
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError

router = APIRouter(prefix="/v1/properties", tags=["properties"])


@router.post(
    "/normalize",
    response_model=NormalizedAddress,
    status_code=status.HTTP_200_OK,
)
def normalize_address(
    payload: AddressPayload,
    brain: Brain = Depends(get_brain),
    settings: Settings = Depends(get_settings),
) -> NormalizedAddress:
    try:
        validate_address_payload(payload, settings)
        return brain.normalize_address(payload)
    except RequestGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except ProviderExecutionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
