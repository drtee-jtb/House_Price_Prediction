# Backend Service Plan

## Goal

Build a production-oriented FastAPI backend that accepts an address, orchestrates external data collection and feature assembly, uses the trained model as the prediction engine, and persists each request as the system of record.

This backend should become the central brain for the project:

- one source of truth for request state, provider results, normalized features, predictions, and model version
- one orchestration path for address parsing, enrichment, validation, prediction, and response shaping
- one internal contract layer that keeps frontend, backend, and data integrations aligned

## Current Implementation Status

The repository now has a working backend foundation, not just a plan.

- FastAPI app and routers are implemented for health, normalization, prediction create, and prediction fetch.
- A central `PredictionOrchestrator` is the single request path for normalization, enrichment, reuse checks, inference, and persistence.
- SQLAlchemy persistence and Alembic migrations exist for requests, canonical normalized addresses, provider responses, feature snapshots, predictions, and model registry records.
- Repeated predictions can reuse recent completed results for the same normalized address and model version.
- Provider calls now run behind timeout and retry wrappers with structured upstream failure mapping.
- Free no-key geocoding and enrichment paths now exist with fallbacks:
  - Nominatim geocoding
  - US Census geocoding
  - US Census tract-level housing context enrichment
  - deterministic fake fallback providers when public upstreams fail
- Geocoding responses and enrichment responses are both persisted as provider snapshots.
- UI-facing read models now exist for recent predictions, prediction detail, and dedicated prediction trace retrieval.
- A compact dashboard bootstrap contract now exposes runtime metadata, provider policy, recent predictions, and route templates for UI initialization.
- The backend can now export an OpenAPI schema artifact for frontend integration.

## Current Starting Point

The repository already has a useful ML core:

- training pipeline in `src/house_price_prediction/model.py`
- preprocessing in `src/house_price_prediction/features.py`
- batch inference helper in `src/house_price_prediction/predict.py`

What is missing is everything around it:

- request and response contracts
- API layer
- orchestration services
- persistence and audit trail
- external data provider adapters
- async job execution and observability

## Recommended Architecture

Use a modular monolith first. That is the fastest path to a professional backend without paying early microservice complexity.

### Layers

1. API layer
   - FastAPI routers
   - request validation
   - auth, rate limiting, idempotency
   - response serialization

2. Application layer
   - orchestration use cases
   - central decision-making logic
   - workflow coordination across providers, feature builders, and model runtime

3. Domain layer
   - business entities such as `PropertyRequest`, `PropertyRecord`, `FeatureVector`, `PredictionResult`
   - status model and lifecycle rules
   - contract definitions for internal communication

4. Infrastructure layer
   - database repositories
   - external provider clients
   - model artifact loader
   - background job queue
   - logging, metrics, tracing

### Central Brain

Create a single orchestration service that is the authoritative path for every prediction request.

Suggested responsibilities:

- accept an address or structured property request
- normalize and geocode the address
- collect third-party or internal property data
- assemble the model-ready feature vector
- validate completeness and confidence
- invoke the prediction runtime
- persist raw inputs, derived features, prediction, and metadata
- return a stable response contract to the caller

This service should own workflow state transitions such as:

- `received`
- `normalized`
- `enriched`
- `features_built`
- `predicted`
- `completed`
- `failed`

## Suggested Project Structure

Keep the current ML package and add a backend package beside it:

```text
src/
  house_price_prediction/
    api/
      main.py
      routers/
        health.py
        predictions.py
        properties.py
    application/
      services/
        prediction_orchestrator.py
        property_enrichment_service.py
        feature_assembly_service.py
      schemas/
        commands.py
        events.py
    domain/
      models/
        property.py
        prediction.py
      contracts/
        prediction_contracts.py
    infrastructure/
      db/
        models.py
        repositories.py
        session.py
      providers/
        geocoding_client.py
        property_data_client.py
      model_runtime/
        registry.py
        predictor.py
      queue/
        workers.py
    config.py
    telemetry.py
```

## Internal Contract System

Define internal contracts with Pydantic models. These contracts are the boundary between API, orchestration, provider adapters, and model runtime.

### Core Command Contracts

```python
class PredictionRequestCommand(BaseModel):
    request_id: UUID
    submitted_at: datetime
    address_line_1: str
    address_line_2: str | None = None
    city: str
    state: str
    postal_code: str
    country: str = "US"
    requested_by: str | None = None
    correlation_id: UUID
```

```python
class EnrichmentCommand(BaseModel):
    request_id: UUID
    normalized_address: NormalizedAddress
    provider_targets: list[str]
```

```python
class FeatureAssemblyCommand(BaseModel):
    request_id: UUID
    normalized_address: NormalizedAddress
    provider_payloads: dict[str, Any]
```

### Core Result Contracts

```python
class FeatureVectorContract(BaseModel):
    request_id: UUID
    model_name: str
    model_version: str
    features: dict[str, float | int | str | bool | None]
    completeness_score: float
```

```python
class PredictionResultContract(BaseModel):
    request_id: UUID
    prediction_id: UUID
    predicted_price: float
    currency: str = "USD"
    confidence_score: float | None = None
    model_name: str
    model_version: str
    feature_snapshot_id: UUID
    generated_at: datetime
```

### Event Contracts

Use explicit events internally, even if you stay inside one service:

- `PredictionRequested`
- `AddressNormalized`
- `PropertyEnrichmentCompleted`
- `FeatureVectorCreated`
- `PredictionCompleted`
- `PredictionFailed`

This gives you clean auditability and a future path to async processing if traffic or provider latency grows.

## API Design

Start with a narrow API surface.

### Public Endpoints

- `POST /v1/predictions`
  - accepts address payload
  - creates a prediction request
  - either returns sync result or `202 Accepted` for async flow

- `GET /v1/predictions/{prediction_id}`
  - returns prediction result, status, model metadata, and source trace

- `GET /v1/predictions/{prediction_id}/trace`
  - returns reusable lineage, provider summaries, and source prediction trace for UI drill-down

- `POST /v1/properties/normalize`
  - useful for UI validation and address confirmation

- `GET /v1/health`
  - liveness, readiness, model availability, provider reachability summary

### Response Shape Principles

- include request id and correlation id in every response
- include model name and model version in prediction responses
- return workflow status for incomplete requests
- separate user-safe messages from internal debug details

## Persistence Model

Use PostgreSQL as the source of truth.

Suggested tables:

- `prediction_requests`
- `normalized_addresses`
- `provider_responses`
- `feature_snapshots`
- `predictions`
- `model_registry`
- `workflow_events`

Persist these categories deliberately:

- original user request
- normalized address
- provider raw payloads
- transformed feature vector
- model version used
- prediction output
- workflow timestamps and failure reasons

That gives you reproducibility, debugging, and retraining data later.

## Data Orchestration Strategy

Treat external providers as unreliable dependencies. Wrap each in an adapter with:

- request timeout
- retry with backoff
- schema normalization
- provider-specific error mapping
- response caching where legally and operationally appropriate

Suggested orchestration flow:

1. Receive request.
2. Validate and normalize address.
3. Check if a recent prediction exists for the same normalized address.
4. Fan out to enrichment providers.
5. Merge provider payloads into one canonical property record.
6. Build model feature vector.
7. Validate required feature coverage.
8. Score with model runtime.
9. Persist artifacts and emit event.
10. Return response.

If provider latency is high, move steps 4 through 9 into a background worker and return a tracking id.

## Model Runtime Strategy

Do not let the API call raw joblib models directly from route handlers.

Introduce a dedicated model runtime component responsible for:

- loading the active artifact once at startup
- exposing a stable `predict(feature_vector)` interface
- attaching model metadata to every prediction
- allowing versioned model rollout later

Longer term, this can evolve into a separate inference service if needed, but not yet.

## Recommended Tech Stack

- FastAPI for HTTP layer
- Pydantic v2 for contracts
- SQLAlchemy 2.x for persistence
- Alembic for migrations
- PostgreSQL for system of record
- httpx for provider clients
- Redis optionally for caching and job coordination
- Celery, Dramatiq, or RQ if async background work becomes necessary
- structlog or standard logging with JSON output
- pytest for service and contract tests

## Security and Professional Standards

Build these in from the start:

- API key or JWT auth depending on caller model
- request id and correlation id propagation
- rate limiting on prediction endpoints
- idempotency key support for duplicate submit protection
- input validation and provider output sanitization
- secrets in environment or secret manager only
- audit trail for every prediction request

## Delivery Roadmap

### Phase 1: Foundation

- add FastAPI app skeleton
- add config management for API and database settings
- add health endpoint
- define Pydantic contracts
- add PostgreSQL and migration setup

### Phase 2: Central Brain

- implement `PredictionOrchestrator`
- add address normalization service
- add repository layer for request and result persistence
- wrap current sklearn inference into a model runtime abstraction

### Phase 3: Data Enrichment

- add provider adapters for geocoding and property data
- add canonical property record builder
- add feature assembly and validation layer
- define provider fallback behavior

### Phase 4: Async and Resilience

- move long-running orchestration to background jobs if needed
- add retries, circuit breaking, and caching
- add structured logging and metrics

### Phase 5: Production Readiness

- contract tests and end-to-end tests
- model registry/versioning
- dashboards and alerting
- containerization and deployment pipeline

## First Build Slice

The smartest first slice is not full provider integration. Build this first:

1. `POST /v1/predictions` with a typed request payload
2. database persistence for request and result
3. a fake enrichment adapter returning deterministic sample property features
4. orchestration service that turns those features into a prediction
5. stable response contract with request id, prediction, and model version

That gives the team a working backend spine before external integrations add noise.

## Immediate Next Tasks for This Repo

1. Add FastAPI, Pydantic, SQLAlchemy, Alembic, Uvicorn, and httpx dependencies.
2. Create the new backend package structure under `src/house_price_prediction/api`, `application`, `domain`, and `infrastructure`.
3. Introduce a `PredictionOrchestrator` service and route all prediction requests through it.
4. Add database models for `prediction_requests`, `feature_snapshots`, and `predictions`.
5. Wrap the current sklearn model in a runtime adapter instead of calling it from scripts only.
6. Add contract tests for request and response schemas before wiring real providers.

