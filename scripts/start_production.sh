#!/usr/bin/env bash
# Production startup script: run Alembic migrations then start uvicorn.
# Render free tier may not run the buildCommand on every restart, so we
# ensure schema is current before the API accepts traffic.
set -e

echo "==> Running Alembic migrations..."
alembic upgrade head

echo "==> Starting API..."
exec uvicorn house_price_prediction.api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --log-level info \
    --timeout-keep-alive 65
