"""Export the FastAPI OpenAPI schema to a JSON file.

Usage:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --output docs/openapi.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema to JSON.")
    parser.add_argument(
        "--output",
        default="docs/openapi.json",
        help="Output path for the schema file (default: docs/openapi.json).",
    )
    args = parser.parse_args()

    # Ensure the app can be imported cleanly with a test-mode environment.
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("TARGET_COLUMN", "SalePrice")

    try:
        from house_price_prediction.api.main import create_app
        from house_price_prediction.config import load_settings

        load_settings.cache_clear()
        app = create_app()
        schema = app.openapi()
    except Exception as exc:
        print(f"ERROR: failed to generate OpenAPI schema: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    route_count = len(schema.get("paths", {}))
    print(f"Exported OpenAPI schema ({route_count} paths) → {out_path}")


if __name__ == "__main__":
    main()
