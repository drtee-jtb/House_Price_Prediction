from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from house_price_prediction.application.contracts.orchestration_contracts import (
    NormalizationStageResult,
    PredictionWorkflowCommand,
    PredictionWorkflowResult,
    ReuseStageResult,
    WorkflowEventContract,
)
from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)
from house_price_prediction.application.services.property_enrichment_service import (
    PropertyEnrichmentService,
)
from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressBaselineResponse,
    AddressPayload,
    BaselineAssessment,
    BaselineCheckResult,
    BaselineExpectationsInput,
    BaselineFeatureObservation,
    BaselineLocationObservation,
    BaselineValueObservation,
    FeaturePolicySimulationItem,
    FeaturePolicySimulationResponse,
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.db.repositories import PredictionRepository
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime
from house_price_prediction.infrastructure.providers.base import GeocodingProvider
from house_price_prediction.infrastructure.providers.resilient import ProviderExecutionError
from house_price_prediction.telemetry import get_logger

logger = get_logger(__name__)


class DataOrchestrationLayer:
    """Coordinates data retrieval, transformation, prediction, and persistence for a request."""

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
        return self._normalize(payload).normalized_address

    def simulate_feature_policies(
        self,
        payload: AddressPayload,
        policy_names: list[str] | None = None,
    ) -> FeaturePolicySimulationResponse:
        normalization_result = self._normalize(payload)
        provider_response = self._property_enrichment_service.build_property_record(
            normalization_result.normalized_address
        )

        selected_policy_names = policy_names or list(self._feature_assembly_service.available_policy_names())
        available_policy_names = set(self._feature_assembly_service.available_policy_names())
        unique_policy_names: list[str] = []
        for policy_name in selected_policy_names:
            normalized_name = policy_name.strip().lower()
            if normalized_name and normalized_name not in unique_policy_names:
                unique_policy_names.append(normalized_name)

        unknown_policy_names = [
            policy_name for policy_name in unique_policy_names if policy_name not in available_policy_names
        ]
        if unknown_policy_names:
            raise ValueError(
                "Unsupported feature policy name(s): " + ", ".join(sorted(unknown_policy_names))
            )

        simulations: list[FeaturePolicySimulationItem] = []
        for policy_name in unique_policy_names:
            feature_vector = self._feature_assembly_service.assemble(
                request_id=uuid4(),
                provider_payload=provider_response.payload,
                context={"state": normalization_result.normalized_address.state},
                policy_name_override=policy_name,
            )
            predicted_price = round(
                self._prediction_runtime.predict(feature_vector.features),
                2,
            )
            simulations.append(
                FeaturePolicySimulationItem(
                    policy_name=feature_vector.feature_policy_name or policy_name,
                    policy_version=feature_vector.feature_policy_version
                    or self._feature_assembly_service.feature_policy_version,
                    predicted_price=predicted_price,
                    completeness_score=feature_vector.completeness_score,
                    weight_total=feature_vector.weight_total,
                )
            )

        return FeaturePolicySimulationResponse(
            normalized_address=normalization_result.normalized_address,
            provider_name=provider_response.provider_name,
            simulations=simulations,
        )

    def generate_address_baseline(
        self,
        payload: AddressPayload,
        expectations: BaselineExpectationsInput | None = None,
    ) -> AddressBaselineResponse:
        normalization_result = self._normalize(payload)
        provider_response = self._property_enrichment_service.build_property_record(
            normalization_result.normalized_address
        )

        feature_vector = self._feature_assembly_service.assemble(
            request_id=uuid4(),
            provider_payload=provider_response.payload,
            context={"state": normalization_result.normalized_address.state},
        )
        predicted_price = round(
            self._prediction_runtime.predict(feature_vector.features),
            2,
        )

        expected_features = list(self._feature_assembly_service.expected_feature_names)
        pulled_features = [
            feature_name
            for feature_name in expected_features
            if feature_vector.features.get(feature_name) is not None
        ]
        missing_features = [
            feature_name
            for feature_name in expected_features
            if feature_vector.features.get(feature_name) is None
        ]

        geocoding_payload = normalization_result.geocoding_provider_response.payload
        geocoding_payload_preview = (
            geocoding_payload
            if len(geocoding_payload) <= 20
            else dict(list(geocoding_payload.items())[:20])
        )

        property_payload = provider_response.payload

        feature_values = {
            name: feature_vector.features.get(name)
            for name in expected_features
        }
        key_feature_names = [
            "BedroomAbvGr",
            "TotRmsAbvGrd",
            "GrLivArea",
            "LotArea",
            "FullBath",
            "HalfBath",
        ]
        key_feature_values = {
            name: feature_values.get(name)
            for name in key_feature_names
        }

        checks: list[BaselineCheckResult] = []
        overall_status = "not_evaluated"
        if expectations is not None:
            expectation_count = 0

            # Only evaluate the completeness threshold when a threshold is explicitly provided.
            # Without this guard, comparing a float score against None raises TypeError.
            if expectations.min_completeness_score is not None:
                expectation_count += 1
                completeness_status = (
                    "passed"
                    if feature_vector.completeness_score >= expectations.min_completeness_score
                    else "failed"
                )
                checks.append(
                    BaselineCheckResult(
                        check_name="completeness_threshold",
                        status=completeness_status,
                        message=(
                            f"Expected completeness >= {expectations.min_completeness_score}, "
                            f"observed {feature_vector.completeness_score:.4f}"
                        ),
                    )
                )

            for required_feature in expectations.required_features:
                expectation_count += 1
                observed_value = feature_values.get(required_feature)
                checks.append(
                    BaselineCheckResult(
                        check_name=f"required_feature:{required_feature}",
                        status="passed" if observed_value is not None else "failed",
                        message=(
                            f"Feature {required_feature} is present"
                            if observed_value is not None
                            else f"Feature {required_feature} is missing"
                        ),
                    )
                )

            for feature_name, bounds in expectations.feature_bounds.items():
                expectation_count += 1
                observed_value = feature_values.get(feature_name)
                status = "failed"
                if isinstance(observed_value, (int, float)):
                    status = (
                        "passed"
                        if bounds.minimum <= float(observed_value) <= bounds.maximum
                        else "failed"
                    )
                checks.append(
                    BaselineCheckResult(
                        check_name=f"feature_bounds:{feature_name}",
                        status=status,
                        message=(
                            f"Expected {feature_name} between {bounds.minimum} and {bounds.maximum}, "
                            f"observed {observed_value}"
                        ),
                    )
                )

            if expectation_count > 0:
                overall_status = "failed" if any(item.status == "failed" for item in checks) else "passed"
            else:
                checks.append(
                    BaselineCheckResult(
                        check_name="no_expectations",
                        status="skipped",
                        message="No expectations were provided; baseline returned raw live observations only.",
                    )
                )

        return AddressBaselineResponse(
            contract_version="2026-04-16",
            evaluated_at=datetime.now(UTC),
            input_address=payload,
            location=BaselineLocationObservation(
                normalized_address=normalization_result.normalized_address,
                geocoding_provider=normalization_result.geocoding_provider_response.provider_name,
                geocoding_payload_preview=geocoding_payload_preview,
            ),
            features=BaselineFeatureObservation(
                expected_features=expected_features,
                pulled_features=pulled_features,
                missing_features=missing_features,
                pulled_feature_count=len(pulled_features),
                missing_feature_count=len(missing_features),
                completeness_score=feature_vector.completeness_score,
                weight_total=feature_vector.weight_total,
                selected_feature_policy_name=feature_vector.feature_policy_name,
                selected_feature_policy_version=feature_vector.feature_policy_version,
                feature_values=feature_values,
                key_feature_values=key_feature_values,
            ),
            value=BaselineValueObservation(
                predicted_price=predicted_price,
                model_name=self._prediction_runtime.model_name,
                model_version=self._prediction_runtime.model_version,
                property_provider=provider_response.provider_name,
                property_feature_source=property_payload.get("feature_source"),
                property_feature_provenance=property_payload.get("feature_provenance"),
            ),
            assessment=BaselineAssessment(
                overall_status=overall_status,
                checks=checks,
            ),
        )

    def execute_prediction_workflow(
        self,
        command: PredictionWorkflowCommand,
    ) -> PredictionWorkflowResult:
        normalization_result = self._normalize(command.payload)
        selected_policy_name = self._feature_assembly_service.resolve_policy_for_context(
            {"state": normalization_result.normalized_address.state}
        )
        selected_policy_version = self._feature_assembly_service.feature_policy_version

        with self._session_factory() as session:
            repository = PredictionRepository(session)
            reuse_stage = self._initialize_request_tracking(
                repository=repository,
                command=command,
                normalization_result=normalization_result,
                selected_feature_policy_name=selected_policy_name,
                selected_feature_policy_version=selected_policy_version,
            )
            self._record_workflow_event(
                repository=repository,
                event=WorkflowEventContract(
                    request_id=command.request_id,
                    event_name="prediction_received",
                    payload={
                        "correlation_id": str(command.correlation_id),
                    },
                    occurred_at=command.submitted_at,
                ),
            )
            self._record_workflow_event(
                repository=repository,
                event=WorkflowEventContract(
                    request_id=command.request_id,
                    event_name="address_normalized",
                    payload={
                        "formatted_address": normalization_result.normalized_address.formatted_address,
                        "geocoding_source": normalization_result.normalized_address.geocoding_source,
                    },
                    occurred_at=datetime.now(UTC),
                ),
            )
            self._record_workflow_event(
                repository=repository,
                event=WorkflowEventContract(
                    request_id=command.request_id,
                    event_name="reuse_candidate_evaluated",
                    payload={
                        "was_reused": reuse_stage.reusable_prediction is not None,
                        "selected_feature_policy_name": selected_policy_name,
                        "selected_feature_policy_version": selected_policy_version,
                    },
                    occurred_at=datetime.now(UTC),
                ),
            )
            session.commit()

            try:
                repository.create_provider_response(
                    request_id=command.request_id,
                    provider_response=normalization_result.geocoding_provider_response,
                )
                session.commit()

                reusable_prediction = reuse_stage.reusable_prediction
                if reusable_prediction is not None:
                    prediction_id = repository.create_prediction(
                        request_id=command.request_id,
                        model_registry_id=reuse_stage.model_registry_id,
                        feature_snapshot_id=reusable_prediction.feature_snapshot_id,
                        predicted_price=reusable_prediction.predicted_price,
                        model_name=reusable_prediction.model_name,
                        model_version=reusable_prediction.model_version,
                        confidence_score=reusable_prediction.confidence_score,
                        was_reused=True,
                        source_prediction_id=reusable_prediction.prediction_id,
                    )
                    repository.update_request_status(command.request_id, status="completed")
                    session.commit()
                    logger.info("prediction_request_reused request_id=%s", command.request_id)
                    self._record_workflow_event(
                        repository=repository,
                        event=WorkflowEventContract(
                            request_id=command.request_id,
                            event_name="prediction_completed",
                            payload={
                                "prediction_id": str(prediction_id),
                                "was_reused": True,
                                "source_prediction_id": str(reusable_prediction.prediction_id),
                                "selected_feature_policy_name": reusable_prediction.selected_feature_policy_name,
                                "selected_feature_policy_version": reusable_prediction.selected_feature_policy_version,
                            },
                            occurred_at=datetime.now(UTC),
                        ),
                    )
                    session.commit()
                    return PredictionWorkflowResult(
                        request_id=command.request_id,
                        prediction_id=prediction_id,
                        correlation_id=command.correlation_id,
                        status="completed",
                        predicted_price=reusable_prediction.predicted_price,
                        currency=reusable_prediction.currency,
                        confidence_score=reusable_prediction.confidence_score,
                        model_name=reusable_prediction.model_name,
                        model_version=reusable_prediction.model_version,
                        feature_snapshot_id=reusable_prediction.feature_snapshot_id,
                        normalized_address=normalization_result.normalized_address,
                        submitted_at=command.submitted_at,
                        generated_at=datetime.now(UTC),
                        was_reused=True,
                        source_prediction_id=reusable_prediction.prediction_id,
                        selected_feature_policy_name=reusable_prediction.selected_feature_policy_name,
                        selected_feature_policy_version=reusable_prediction.selected_feature_policy_version,
                    )

                provider_response = self._property_enrichment_service.build_property_record(
                    normalization_result.normalized_address
                )
                repository.create_provider_response(
                    request_id=command.request_id,
                    provider_response=provider_response,
                )
                self._record_workflow_event(
                    repository=repository,
                    event=WorkflowEventContract(
                        request_id=command.request_id,
                        event_name="property_enrichment_completed",
                        payload={
                            "provider_name": provider_response.provider_name,
                            "status": provider_response.status,
                        },
                        occurred_at=provider_response.fetched_at,
                    ),
                )

                feature_vector = self._feature_assembly_service.assemble(
                    request_id=command.request_id,
                    provider_payload=provider_response.payload,
                    context={"state": normalization_result.normalized_address.state},
                )
                self._record_workflow_event(
                    repository=repository,
                    event=WorkflowEventContract(
                        request_id=command.request_id,
                        event_name="feature_vector_created",
                        payload={
                            "completeness_score": feature_vector.completeness_score,
                            "feature_count": len(feature_vector.features),
                            "selected_feature_policy_name": feature_vector.feature_policy_name,
                            "selected_feature_policy_version": feature_vector.feature_policy_version,
                        },
                        occurred_at=datetime.now(UTC),
                    ),
                )
                prediction_value = round(
                    self._prediction_runtime.predict(feature_vector.features),
                    2,
                )
                feature_snapshot_id = repository.create_feature_snapshot(feature_vector)
                prediction_id = repository.create_prediction(
                    request_id=command.request_id,
                    model_registry_id=reuse_stage.model_registry_id,
                    feature_snapshot_id=feature_snapshot_id,
                    predicted_price=prediction_value,
                    model_name=self._prediction_runtime.model_name,
                    model_version=self._prediction_runtime.model_version,
                    confidence_score=None,
                )
                repository.update_request_status(command.request_id, status="completed")
                self._record_workflow_event(
                    repository=repository,
                    event=WorkflowEventContract(
                        request_id=command.request_id,
                        event_name="prediction_completed",
                        payload={
                            "prediction_id": str(prediction_id),
                            "was_reused": False,
                            "selected_feature_policy_name": feature_vector.feature_policy_name,
                            "selected_feature_policy_version": feature_vector.feature_policy_version,
                        },
                        occurred_at=datetime.now(UTC),
                    ),
                )
                session.commit()
                logger.info("prediction_request_completed request_id=%s", command.request_id)
            except Exception as exc:
                session.rollback()

                if isinstance(exc, ProviderExecutionError):
                    repository.create_provider_response(
                        request_id=command.request_id,
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
                    command.request_id,
                    status="failed",
                    error_message=str(exc),
                )
                self._record_workflow_event(
                    repository=repository,
                    event=WorkflowEventContract(
                        request_id=command.request_id,
                        event_name="prediction_failed",
                        payload={
                            "error": str(exc),
                        },
                        occurred_at=datetime.now(UTC),
                    ),
                )
                session.commit()
                logger.warning(
                    "prediction_request_failed request_id=%s error=%s",
                    command.request_id,
                    exc,
                )
                raise

        return PredictionWorkflowResult(
            request_id=command.request_id,
            prediction_id=prediction_id,
            correlation_id=command.correlation_id,
            status="completed",
            predicted_price=prediction_value,
            currency="USD",
            confidence_score=None,
            model_name=self._prediction_runtime.model_name,
            model_version=self._prediction_runtime.model_version,
            feature_snapshot_id=feature_snapshot_id,
            normalized_address=normalization_result.normalized_address,
            submitted_at=command.submitted_at,
            generated_at=datetime.now(UTC),
            was_reused=False,
            source_prediction_id=None,
            selected_feature_policy_name=feature_vector.feature_policy_name,
            selected_feature_policy_version=feature_vector.feature_policy_version,
        )

    def _normalize(self, payload: AddressPayload) -> NormalizationStageResult:
        geocoding_result = self._geocoding_provider.normalize(payload)
        return NormalizationStageResult(
            normalized_address=geocoding_result.normalized_address,
            geocoding_provider_response=geocoding_result.provider_response,
        )

    def _initialize_request_tracking(
        self,
        repository: PredictionRepository,
        command: PredictionWorkflowCommand,
        normalization_result: NormalizationStageResult,
        selected_feature_policy_name: str,
        selected_feature_policy_version: str,
    ) -> ReuseStageResult:
        normalized_address_id = repository.get_or_create_normalized_address(
            normalization_result.normalized_address
        )
        model_registry_id = repository.register_model_version(
            model_name=self._prediction_runtime.model_name,
            model_version=self._prediction_runtime.model_version,
            feature_columns=list(self._prediction_runtime.expected_feature_names()),
        )
        repository.create_prediction_request(
            request_id=command.request_id,
            correlation_id=command.correlation_id,
            normalized_address_id=normalized_address_id,
            payload=command.payload,
            normalized_address=normalization_result.normalized_address,
            submitted_at=command.submitted_at,
            feature_policy_name=selected_feature_policy_name,
            feature_policy_version=selected_feature_policy_version,
        )

        reusable_prediction = repository.find_reusable_prediction(
            normalized_address_id=normalized_address_id,
            model_registry_id=model_registry_id,
            max_age_hours=self._prediction_reuse_max_age_hours,
            feature_policy_name=selected_feature_policy_name,
            feature_policy_version=selected_feature_policy_version,
        )

        return ReuseStageResult(
            normalized_address_id=normalized_address_id,
            model_registry_id=model_registry_id,
            reusable_prediction=reusable_prediction,
        )

    @staticmethod
    def _record_workflow_event(
        repository: PredictionRepository,
        event: WorkflowEventContract,
    ) -> None:
        repository.create_workflow_event(
            request_id=event.request_id,
            event_name=event.event_name,
            payload=event.payload,
            occurred_at=event.occurred_at,
        )