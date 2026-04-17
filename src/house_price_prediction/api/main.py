from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from house_price_prediction.api.routers import (
    dashboard,
    health,
    policies,
    predictions,
    properties,
    validation,
)
from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)
from house_price_prediction.application.services.prediction_orchestrator import (
    Brain,
)
from house_price_prediction.application.services.property_enrichment_service import (
    PropertyEnrichmentService,
)
from house_price_prediction.config import Settings, load_settings
from house_price_prediction.infrastructure.db.session import init_database
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime
from house_price_prediction.infrastructure.providers.factory import (
    create_geocoding_provider,
    create_property_data_provider,
)


def create_app(
    settings: Settings | None = None,
    property_data_provider=None,
    geocoding_provider=None,
) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session_factory = init_database(
            app_settings.database_url,
            create_schema=app_settings.app_env.strip().lower() == "test",
            validate_schema=app_settings.app_env.strip().lower() != "test",
        )
        prediction_runtime = PredictionRuntime(app_settings)
        selected_property_data_provider = property_data_provider or create_property_data_provider(
            app_settings
        )
        selected_geocoding_provider = geocoding_provider or create_geocoding_provider(app_settings)
        enrichment_service = PropertyEnrichmentService(selected_property_data_provider)
        feature_assembly_service = FeatureAssemblyService(
            model_name=app_settings.model_name,
            model_version=app_settings.model_version,
            expected_feature_names=prediction_runtime.expected_feature_names(),
            feature_policy_name=app_settings.feature_policy_name,
            feature_policy_version=app_settings.feature_policy_version,
            feature_policy_state_overrides=app_settings.feature_policy_state_overrides or {},
        )

        app.state.settings = app_settings
        app.state.session_factory = session_factory
        app.state.prediction_runtime = prediction_runtime
        app.state.brain = Brain(
            session_factory=session_factory,
            feature_assembly_service=feature_assembly_service,
            prediction_runtime=prediction_runtime,
            property_enrichment_service=enrichment_service,
            geocoding_provider=selected_geocoding_provider,
            prediction_reuse_max_age_hours=app_settings.prediction_reuse_max_age_hours,
            settings=app_settings,
        )
        yield

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.model_version,
        lifespan=lifespan,
    )
    app.include_router(dashboard.router)
    app.include_router(health.router)
    app.include_router(policies.router)
    app.include_router(predictions.router)
    app.include_router(properties.router)
    app.include_router(validation.router)
    return app


app = create_app()