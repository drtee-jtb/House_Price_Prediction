from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from house_price_prediction.api.dependencies import get_brain
from house_price_prediction.application.services.prediction_orchestrator import Brain
from house_price_prediction.domain.contracts.prediction_contracts import (
    ApiCapabilitiesResponse,
    LiveFeatureCandidatesResponse,
)

router = APIRouter(prefix="/v1/meta", tags=["meta"])


@router.get("/capabilities", response_model=ApiCapabilitiesResponse)
def get_api_capabilities(
    brain: Brain = Depends(get_brain),
) -> ApiCapabilitiesResponse:
    return brain.get_api_capabilities()


@router.get("/live-feature-candidates", response_model=LiveFeatureCandidatesResponse)
def get_live_feature_candidates(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_completeness_score: float = Query(default=0.0, ge=0.0, le=1.0),
    include_reused: bool = Query(default=False),
    brain: Brain = Depends(get_brain),
) -> LiveFeatureCandidatesResponse:
    return brain.get_live_feature_candidates(
        limit=limit,
        offset=offset,
        min_completeness_score=min_completeness_score,
        include_reused=include_reused,
    )
