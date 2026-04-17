from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from house_price_prediction.api.dependencies import get_brain
from house_price_prediction.application.services.prediction_orchestrator import (
    Brain,
)
from house_price_prediction.domain.contracts.prediction_contracts import (
    DashboardBootstrapResponse,
)

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


@router.get("/bootstrap", response_model=DashboardBootstrapResponse)
def get_dashboard_bootstrap(
    limit: int = Query(default=5, ge=1, le=20),
    orchestrator: Brain = Depends(get_brain),
) -> DashboardBootstrapResponse:
    return orchestrator.get_dashboard_bootstrap(limit=limit)
