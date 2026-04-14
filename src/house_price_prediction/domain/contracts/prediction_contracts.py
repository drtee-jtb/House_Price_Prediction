from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AddressPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str = Field(min_length=1, max_length=120)
    address_line_2: str | None = Field(default=None, max_length=120)
    city: str = Field(min_length=1, max_length=80)
    state: str = Field(min_length=2, max_length=30)
    postal_code: str = Field(min_length=3, max_length=20)
    country: str = Field(default="US", min_length=2, max_length=60)


class PredictionRequestPayload(AddressPayload):
    requested_by: str | None = Field(default=None, max_length=120)


class NormalizedAddress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str
    address_line_2: str | None = None
    city: str
    state: str
    postal_code: str
    country: str
    formatted_address: str
    latitude: float | None = None
    longitude: float | None = None
    geocoding_source: str | None = None


class FeatureVectorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    model_name: str
    model_version: str
    features: dict[str, Any]
    completeness_score: float


class ProviderResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    payload: dict[str, Any]
    fetched_at: datetime


class ProviderResponseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    fetched_at: datetime
    feature_source: str | None = None
    feature_provenance: dict[str, Any] | None = None
    payload_preview: dict[str, Any]


class FeatureSnapshotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_snapshot_id: UUID
    model_name: str
    model_version: str
    completeness_score: float
    feature_count: int
    populated_feature_count: int
    features: dict[str, Any]


class GeocodingResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    provider_response: ProviderResponseContract


PredictionStatus = Literal["received", "completed", "failed"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    correlation_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    confidence_score: float | None = None
    model_name: str
    model_version: str
    feature_snapshot_id: UUID
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None


class PredictionDetailResponse(PredictionResponse):
    feature_snapshot: FeatureSnapshotSummary
    provider_responses: list[ProviderResponseSummary]
    error_message: str | None = None


class PredictionListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    model_name: str
    model_version: str
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None
    requested_by: str | None = None
    feature_source: str | None = None


class PredictionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionListItem]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    environment: str
    app_name: str
    model_name: str
    model_version: str
    model_available: bool
    mock_predictor_enabled: bool
    property_data_provider: str
    geocoding_provider: str
    provider_timeout_seconds: float
    provider_max_retries: int
    prediction_reuse_max_age_hours: int