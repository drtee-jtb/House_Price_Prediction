from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from house_price_prediction.domain.contracts.prediction_contracts import (
    FeatureSnapshotSummary,
    FeatureVectorContract,
    NormalizedAddress,
    PredictionDetailResponse,
    PredictionListItem,
    PredictionListResponse,
    PredictionRequestPayload,
    ProviderResponseContract,
    ProviderResponseSummary,
)
from house_price_prediction.infrastructure.db.models import (
    FeatureSnapshotModel,
    ModelRegistryModel,
    NormalizedAddressModel,
    PredictionModel,
    PredictionRequestModel,
    ProviderResponseModel,
)


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_prediction_request(
        self,
        request_id: UUID,
        correlation_id: UUID,
        normalized_address_id: UUID,
        payload: PredictionRequestPayload,
        normalized_address: NormalizedAddress,
        submitted_at: datetime,
    ) -> None:
        self._session.add(
            PredictionRequestModel(
                id=str(request_id),
                correlation_id=str(correlation_id),
                normalized_address_id=str(normalized_address_id),
                requested_by=payload.requested_by,
                address_line_1=payload.address_line_1,
                address_line_2=payload.address_line_2,
                city=payload.city,
                state=payload.state,
                postal_code=payload.postal_code,
                country=payload.country,
                normalized_address=normalized_address.formatted_address,
                status="received",
                submitted_at=submitted_at,
            )
        )

    def get_or_create_normalized_address(self, normalized_address: NormalizedAddress) -> UUID:
        existing = self._session.scalar(
            select(NormalizedAddressModel).where(
                NormalizedAddressModel.formatted_address == normalized_address.formatted_address
            )
        )
        if existing is not None:
            if normalized_address.latitude is not None:
                existing.latitude = normalized_address.latitude
            if normalized_address.longitude is not None:
                existing.longitude = normalized_address.longitude
            if normalized_address.geocoding_source is not None:
                existing.geocoding_source = normalized_address.geocoding_source
            return UUID(existing.id)

        normalized_address_id = uuid4()
        self._session.add(
            NormalizedAddressModel(
                id=str(normalized_address_id),
                address_line_1=normalized_address.address_line_1,
                address_line_2=normalized_address.address_line_2,
                city=normalized_address.city,
                state=normalized_address.state,
                postal_code=normalized_address.postal_code,
                country=normalized_address.country,
                formatted_address=normalized_address.formatted_address,
                latitude=normalized_address.latitude,
                longitude=normalized_address.longitude,
                geocoding_source=normalized_address.geocoding_source,
            )
        )
        return normalized_address_id

    def register_model_version(
        self,
        model_name: str,
        model_version: str,
        feature_columns: list[str],
    ) -> UUID:
        existing = self._session.scalar(
            select(ModelRegistryModel).where(
                ModelRegistryModel.model_name == model_name,
                ModelRegistryModel.model_version == model_version,
            )
        )
        if existing is not None:
            existing.feature_columns = feature_columns
            existing.is_active = True
            return UUID(existing.id)

        self._session.query(ModelRegistryModel).filter(
            ModelRegistryModel.model_name == model_name
        ).update({ModelRegistryModel.is_active: False}, synchronize_session=False)

        model_registry_id = uuid4()
        self._session.add(
            ModelRegistryModel(
                id=str(model_registry_id),
                model_name=model_name,
                model_version=model_version,
                feature_columns=feature_columns,
                is_active=True,
            )
        )
        return model_registry_id

    def update_request_status(
        self,
        request_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        request = self._session.get(PredictionRequestModel, str(request_id))
        if request is None:
            return
        request.status = status
        request.error_message = error_message

    def create_feature_snapshot(self, feature_vector: FeatureVectorContract) -> UUID:
        feature_snapshot_id = uuid4()
        self._session.add(
            FeatureSnapshotModel(
                id=str(feature_snapshot_id),
                request_id=str(feature_vector.request_id),
                model_name=feature_vector.model_name,
                model_version=feature_vector.model_version,
                completeness_score=feature_vector.completeness_score,
                features=feature_vector.features,
            )
        )
        return feature_snapshot_id

    def create_provider_response(
        self,
        request_id: UUID,
        provider_response: ProviderResponseContract,
    ) -> UUID:
        provider_response_id = uuid4()
        self._session.add(
            ProviderResponseModel(
                id=str(provider_response_id),
                request_id=str(request_id),
                provider_name=provider_response.provider_name,
                status=provider_response.status,
                payload=provider_response.payload,
                fetched_at=provider_response.fetched_at,
            )
        )
        return provider_response_id

    def create_prediction(
        self,
        request_id: UUID,
        model_registry_id: UUID,
        feature_snapshot_id: UUID,
        predicted_price: float,
        model_name: str,
        model_version: str,
        confidence_score: float | None = None,
        was_reused: bool = False,
        source_prediction_id: UUID | None = None,
    ) -> UUID:
        prediction_id = uuid4()
        self._session.add(
            PredictionModel(
                id=str(prediction_id),
                request_id=str(request_id),
                model_registry_id=str(model_registry_id),
                source_prediction_id=str(source_prediction_id) if source_prediction_id else None,
                feature_snapshot_id=str(feature_snapshot_id),
                predicted_price=predicted_price,
                currency="USD",
                confidence_score=confidence_score,
                model_name=model_name,
                model_version=model_version,
                was_reused=was_reused,
            )
        )
        return prediction_id

    def get_prediction_detail(self, prediction_id: UUID) -> PredictionDetailResponse | None:
        prediction = self._session.get(PredictionModel, str(prediction_id))
        if prediction is None:
            return None

        request = self._session.get(PredictionRequestModel, prediction.request_id)
        feature_snapshot = self._session.get(FeatureSnapshotModel, prediction.feature_snapshot_id)
        if request is None or feature_snapshot is None:
            return None

        normalized_address = self._build_normalized_address(request)
        trace_feature_snapshot = feature_snapshot
        trace_request_id = UUID(request.id)
        trace_prediction = self._resolve_trace_prediction(prediction)
        if trace_prediction is not None:
            trace_request_id = UUID(trace_prediction.request_id)
            source_feature_snapshot = self._session.get(
                FeatureSnapshotModel, trace_prediction.feature_snapshot_id
            )
            if source_feature_snapshot is not None:
                trace_feature_snapshot = source_feature_snapshot

        provider_responses = self.get_provider_responses(trace_request_id)
        return PredictionDetailResponse(
            request_id=UUID(request.id),
            prediction_id=UUID(prediction.id),
            correlation_id=UUID(request.correlation_id),
            status=request.status,
            predicted_price=prediction.predicted_price,
            currency=prediction.currency,
            confidence_score=prediction.confidence_score
            if prediction.confidence_score is not None
            else feature_snapshot.completeness_score,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            feature_snapshot_id=UUID(feature_snapshot.id),
            feature_snapshot=self._build_feature_snapshot_summary(trace_feature_snapshot),
            provider_responses=[
                self._build_provider_response_summary(provider_response)
                for provider_response in provider_responses
            ],
            normalized_address=normalized_address,
            submitted_at=request.submitted_at,
            generated_at=prediction.generated_at,
            was_reused=prediction.was_reused,
            source_prediction_id=UUID(prediction.source_prediction_id)
            if prediction.source_prediction_id is not None
            else None,
            error_message=request.error_message,
        )

    def list_recent_predictions(self, limit: int = 10) -> PredictionListResponse:
        statement = (
            select(PredictionModel)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .where(PredictionRequestModel.status == "completed")
            .order_by(PredictionModel.generated_at.desc())
            .limit(limit)
        )
        prediction_rows = self._session.scalars(statement).all()
        items: list[PredictionListItem] = []
        for prediction in prediction_rows:
            request = self._session.get(PredictionRequestModel, prediction.request_id)
            feature_snapshot = self._session.get(FeatureSnapshotModel, prediction.feature_snapshot_id)
            if request is None or feature_snapshot is None:
                continue

            normalized_address = self._build_normalized_address(request)
            trace_request_id = UUID(request.id)
            trace_prediction = self._resolve_trace_prediction(prediction)
            if trace_prediction is not None:
                trace_request_id = UUID(trace_prediction.request_id)

            provider_responses = self.get_provider_responses(trace_request_id)
            feature_source = self._find_feature_source(provider_responses)
            items.append(
                PredictionListItem(
                    request_id=UUID(request.id),
                    prediction_id=UUID(prediction.id),
                    status=request.status,
                    predicted_price=prediction.predicted_price,
                    currency=prediction.currency,
                    model_name=prediction.model_name,
                    model_version=prediction.model_version,
                    normalized_address=normalized_address,
                    submitted_at=request.submitted_at,
                    generated_at=prediction.generated_at,
                    was_reused=prediction.was_reused,
                    source_prediction_id=UUID(prediction.source_prediction_id)
                    if prediction.source_prediction_id is not None
                    else None,
                    requested_by=request.requested_by,
                    feature_source=feature_source,
                )
            )

        return PredictionListResponse(items=items)

    def find_reusable_prediction(
        self,
        normalized_address_id: UUID,
        model_registry_id: UUID,
        max_age_hours: int,
    ) -> PredictionDetailResponse | None:
        if max_age_hours <= 0:
            return None

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        statement = (
            select(PredictionModel.id)
            .join(PredictionRequestModel, PredictionRequestModel.id == PredictionModel.request_id)
            .where(PredictionRequestModel.normalized_address_id == str(normalized_address_id))
            .where(PredictionRequestModel.status == "completed")
            .where(PredictionModel.model_registry_id == str(model_registry_id))
            .where(PredictionModel.generated_at >= cutoff)
            .order_by(PredictionModel.generated_at.desc())
            .limit(1)
        )
        prediction_id = self._session.scalar(statement)
        if prediction_id is None:
            return None
        return self.get_prediction_detail(UUID(prediction_id))

    def get_provider_responses(self, request_id: UUID) -> list[ProviderResponseContract]:
        provider_rows = (
            self._session.query(ProviderResponseModel)
            .filter(ProviderResponseModel.request_id == str(request_id))
            .order_by(ProviderResponseModel.fetched_at.asc())
            .all()
        )
        return [
            ProviderResponseContract(
                provider_name=row.provider_name,
                status=row.status,
                payload=row.payload,
                fetched_at=row.fetched_at,
            )
            for row in provider_rows
        ]

    def get_request_id_by_correlation_id(self, correlation_id: UUID) -> UUID | None:
        request_id = self._session.scalar(
            select(PredictionRequestModel.id)
            .where(PredictionRequestModel.correlation_id == str(correlation_id))
            .order_by(PredictionRequestModel.submitted_at.desc())
            .limit(1)
        )
        return UUID(request_id) if request_id is not None else None

    def get_request_status(self, request_id: UUID) -> str | None:
        request = self._session.get(PredictionRequestModel, str(request_id))
        return request.status if request is not None else None

    def count_normalized_addresses(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(NormalizedAddressModel)) or 0)

    def get_model_registry_entry(self, model_name: str, model_version: str) -> ModelRegistryModel | None:
        return self._session.scalar(
            select(ModelRegistryModel).where(
                ModelRegistryModel.model_name == model_name,
                ModelRegistryModel.model_version == model_version,
            )
        )

    @staticmethod
    def _build_feature_snapshot_summary(
        feature_snapshot: FeatureSnapshotModel,
    ) -> FeatureSnapshotSummary:
        features = dict(feature_snapshot.features)
        feature_count = len(features)
        populated_feature_count = sum(value is not None for value in features.values())
        return FeatureSnapshotSummary(
            feature_snapshot_id=UUID(feature_snapshot.id),
            model_name=feature_snapshot.model_name,
            model_version=feature_snapshot.model_version,
            completeness_score=feature_snapshot.completeness_score,
            feature_count=feature_count,
            populated_feature_count=populated_feature_count,
            features=features,
        )

    @staticmethod
    def _build_provider_response_summary(
        provider_response: ProviderResponseContract,
    ) -> ProviderResponseSummary:
        payload = provider_response.payload
        payload_preview = {
            key: value
            for key, value in payload.items()
            if key not in {"result", "query", "normalized", "features"}
        }
        return ProviderResponseSummary(
            provider_name=provider_response.provider_name,
            status=provider_response.status,
            fetched_at=provider_response.fetched_at,
            feature_source=payload.get("feature_source"),
            feature_provenance=payload.get("feature_provenance"),
            payload_preview=payload_preview,
        )

    @staticmethod
    def _find_feature_source(
        provider_responses: list[ProviderResponseContract],
    ) -> str | None:
        for provider_response in reversed(provider_responses):
            feature_source = provider_response.payload.get("feature_source")
            if isinstance(feature_source, str):
                return feature_source
        return None

    def _resolve_trace_prediction(
        self,
        prediction: PredictionModel,
    ) -> PredictionModel | None:
        current = prediction
        visited_ids = {current.id}
        while current.source_prediction_id is not None:
            source_prediction = self._session.get(PredictionModel, current.source_prediction_id)
            if source_prediction is None or source_prediction.id in visited_ids:
                break
            current = source_prediction
            visited_ids.add(current.id)
        return current

    def _build_normalized_address(self, request: PredictionRequestModel) -> NormalizedAddress:
        normalized_address_row = self._session.get(
            NormalizedAddressModel, request.normalized_address_id
        )
        address_line_1 = " ".join(request.address_line_1.strip().upper().split())
        city = " ".join(request.city.strip().upper().split())
        state = request.state.strip().upper()
        postal_code = request.postal_code.strip().upper()
        country = request.country.strip().upper()
        address_line_2 = (
            " ".join(request.address_line_2.strip().upper().split())
            if request.address_line_2
            else None
        )
        return NormalizedAddress(
            address_line_1=address_line_1,
            address_line_2=address_line_2,
            city=city,
            state=state,
            postal_code=postal_code,
            country=country,
            formatted_address=request.normalized_address,
            latitude=getattr(normalized_address_row, "latitude", None),
            longitude=getattr(normalized_address_row, "longitude", None),
            geocoding_source=getattr(normalized_address_row, "geocoding_source", None),
        )