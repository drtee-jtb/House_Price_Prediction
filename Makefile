.PHONY: setup bootstrap-data train predict test lint clean smoke-api smoke-real ci-verify

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt
	. .venv/bin/activate && pip install -e .

train:
	python scripts/train.py

bootstrap-data:
	python scripts/bootstrap_training_data.py

predict:
	python scripts/predict.py

test:
	pytest

smoke-api:
	python scripts/smoke_api.py

smoke-real:
	rm -f data/processed/real_validation.db
	ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback PREDICTION_REUSE_MAX_AGE_HOURS=0 DATABASE_URL=sqlite:///data/processed/real_validation.db python scripts/smoke_api.py

ci-verify:
	python scripts/bootstrap_training_data.py
	python scripts/train.py
	pytest
	rm -f data/processed/smoke_validation.db
	DATABASE_URL=sqlite:///data/processed/smoke_validation.db PREDICTION_REUSE_MAX_AGE_HOURS=0 python scripts/smoke_api.py
	rm -f data/processed/real_validation.db
	ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback PREDICTION_REUSE_MAX_AGE_HOURS=0 DATABASE_URL=sqlite:///data/processed/real_validation.db python scripts/smoke_api.py

lint:
	python -m py_compile src/house_price_prediction/*.py scripts/*.py

clean:
	rm -rf .pytest_cache
