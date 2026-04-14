from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import sessionmaker

from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)
from house_price_prediction.application.services.property_enrichment_service import (
    PropertyEnrichmentService,
)
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    NormalizedAddress,
    PredictionDetailResponse,
    PredictionListResponse,
    PredictionRequestPayload,
    PredictionResponse,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.db.repositories import PredictionRepository
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime
from house_price_prediction.infrastructure.providers.base import GeocodingProvider
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError
from house_price_prediction.telemetry import correlation_scope, get_logger

logger = get_logger(__name__)


class PredictionOrchestrator:
    def __init__(
        self,
        session_factory: sessionmaker,
        feature_assembly_service: FeatureAssemblyService,
        prediction_runtime: PredictionRuntime,
        property_enrichment_service: PropertyEnrichmentService,
        geocoding_provider: GeocodingProvider,
        prediction_reuse_max_age_hours: int,
    ) -> None:
        self._session_factory = session_factory
        self._feature_assembly_service = feature_assembly_service
        self._prediction_runtime = prediction_runtime
        self._property_enrichment_service = property_enrichment_service
        self._geocoding_provider = geocoding_provider
        self._prediction_reuse_max_age_hours = prediction_reuse_max_age_hours

    def normalize_address(self, payload: AddressPayload) -> NormalizedAddress:
        geocoding_result = self._geocoding_provider.normalize(payload)
        return geocoding_result.normalized_address

    def get_prediction_detail(self, prediction_id: UUID) -> PredictionDetailResponse | None:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.get_prediction_detail(prediction_id)

    def list_recent_predictions(self, limit: int = 10) -> PredictionListResponse:
        with self._session_factory() as session:
            repository = PredictionRepository(session)
            return repository.list_recent_predictions(limit=limit)

    def create_prediction(
        self,
        payload: PredictionRequestPayload,
        correlation_id: str | None = None,
    ) -> PredictionResponse:
        request_id = uuid4()
        correlation_uuid = self._parse_correlation_id(correlation_id)
        submitted_at = datetime.now(UTC)

        with correlation_scope(str(correlation_uuid)):
            logger.info("prediction_request_start request_id=%s", request_id)
            geocoding_result = self._geocoding_provider.normalize(payload)
            normalized_address = geocoding_result.normalized_address

            with self._session_factory() as session:
                repository = PredictionRepository(session)
                normalized_address_id = repository.get_or_create_normalized_address(normalized_address)
                model_registry_id = repository.register_model_version(
                    model_name=self._prediction_runtime.model_name,
                    model_version=self._prediction_runtime.model_version,
                    feature_columns=list(self._prediction_runtime.expected_feature_names()),
                )
                repository.create_prediction_request(
                    request_id=request_id,
                    correlation_id=correlation_uuid,
                    normalized_address_id=normalized_address_id,
                    payload=payload,
                    normalized_address=normalized_address,
                    submitted_at=submitted_at,
                )
                session.commit()

                try:
                    repository.create_provider_response(
                        request_id=request_id,
                        provider_response=geocoding_result.provider_response,
                    )
                    session.commit()

                    reusable_prediction = repository.find_reusable_prediction(
                        normalized_address_id=normalized_address_id,
                        model_registry_id=model_registry_id,
                        max_age_hours=self._prediction_reuse_max_age_hours,
                    )
                    if reusable_prediction is not None:
                        prediction_id = repository.create_prediction(
                            request_id=request_id,
                            model_registry_id=model_registry_id,
                            feature_snapshot_id=reusable_prediction.feature_snapshot_id,
                            predicted_price=reusable_prediction.predicted_price,
                            model_name=reusable_prediction.model_name,
                            model_version=reusable_prediction.model_version,
                            confidence_score=reusable_prediction.confidence_score,
                            was_reused=True,
                            source_prediction_id=reusable_prediction.prediction_id,
                        )
                        repository.update_request_status(request_id, status="completed")
                        session.commit()
                        logger.info("prediction_request_reused request_id=%s", request_id)
                        return PredictionResponse(
                            request_id=request_id,
                            prediction_id=prediction_id,
                            correlation_id=correlation_uuid,
                            status="completed",
                            predicted_price=reusable_prediction.predicted_price,
                            currency=reusable_prediction.currency,
                            confidence_score=reusable_prediction.confidence_score,
                            model_name=reusable_prediction.model_name,
                            model_version=reusable_prediction.model_version,
                            feature_snapshot_id=reusable_prediction.feature_snapshot_id,
                            normalized_address=normalized_address,
                            submitted_at=submitted_at,
                            generated_at=datetime.now(UTC),
                            was_reused=True,
                            source_prediction_id=reusable_prediction.prediction_id,
                        )

                    provider_response = self._property_enrichment_service.build_property_record(
                        normalized_address
                    )
                    repository.create_provider_response(
                        request_id=request_id,
                        provider_response=provider_response,
                    )
                    feature_vector = self._feature_assembly_service.assemble(
                        request_id=request_id,
                        provider_payload=provider_response.payload,
                    )
                    prediction_value = self._prediction_runtime.predict(feature_vector.features)
                    feature_snapshot_id = repository.create_feature_snapshot(feature_vector)
                    prediction_id = repository.create_prediction(
                        request_id=request_id,
                        model_registry_id=model_registry_id,
                        feature_snapshot_id=feature_snapshot_id,
                        predicted_price=prediction_value,
                        model_name=self._prediction_runtime.model_name,
                        model_version=self._prediction_runtime.model_version,
                        confidence_score=feature_vector.completeness_score,
                    )
                    repository.update_request_status(request_id, status="completed")
                    session.commit()
                    logger.info("prediction_request_completed request_id=%s", request_id)
                except Exception as exc:
                    session.rollback()

                    if isinstance(exc, ProviderExecutionError):
                        repository.create_provider_response(
                            request_id=request_id,
                            provider_response=ProviderResponseContract(
                                provider_name=exc.provider_name,
                                status="failed",
                                payload={
                                    "error": str(exc),
                                    "stage": "provider_execution",
                                },
                                fetched_at=datetime.now(UTC),
                            ),
                        )

                    repository.update_request_status(
                        request_id,
                        status="failed",
                        error_message=str(exc),
                    )
                    session.commit()
                    logger.warning("prediction_request_failed request_id=%s error=%s", request_id, exc)
                    raise

            return PredictionResponse(
                request_id=request_id,
                prediction_id=prediction_id,
                correlation_id=correlation_uuid,
                status="completed",
                predicted_price=round(prediction_value, 2),
                currency="USD",
                confidence_score=feature_vector.completeness_score,
                model_name=self._prediction_runtime.model_name,
                model_version=self._prediction_runtime.model_version,
                feature_snapshot_id=feature_snapshot_id,
                normalized_address=normalized_address,
                submitted_at=submitted_at,
                generated_at=datetime.now(UTC),
                was_reused=False,
                source_prediction_id=None,
            )

    @staticmethod
    def _parse_correlation_id(correlation_id: str | None) -> UUID:
        if correlation_id is None:
            return uuid4()
        try:
            return UUID(correlation_id)
        except ValueError:
            return uuid4()