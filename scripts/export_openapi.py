from __future__ import annotations

import json
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from house_price_prediction.api.main import create_app
from house_price_prediction.config import load_settings


if __name__ == "__main__":
    settings = load_settings()
    app = create_app(settings)
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description="House Price Prediction backend API schema for frontend integration.",
    )

    output_path = Path("docs/openapi.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"OpenAPI schema written to: {output_path}")