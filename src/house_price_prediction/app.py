"""
FastAPI backend for House Price Prediction.
Exposes endpoints to predict house prices given an address using free, legal APIs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .address_to_price import PricePredictionPipeline
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices from addresses using County Assessor, Census, and Geocoding APIs",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline once on startup
pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = PricePredictionPipeline()
    logger.info("Pipeline initialized")


class AddressRequest(BaseModel):
    address: str


class PriceResponse(BaseModel):
    address: str
    predicted_price: float
    confidence: float
    error_margin: float
    error_margin_low: float
    error_margin_high: float
    all_16_features: dict
    school_district: str
    school_rating: float
    timestamp: str


@app.get("/")
async def root():
    """API status endpoint."""
    return {
        "status": "running",
        "service": "House Price Prediction API",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PriceResponse)
async def predict_price(request: AddressRequest):
    """
    Predict house price from an address.

    Uses free, legal APIs:
    - Nominatim (OpenStreetMap) for geocoding
    - FCC API for Census tract lookup
    - County Assessor for property data
    - Census data for economic indicators

    Args:
        address: Full address (e.g., "123 Main St, Seattle, WA 98101")

    Returns:
        Price prediction with confidence and feature breakdown
    """
    try:
        # Initialize pipeline if needed
        global pipeline
        if pipeline is None:
            pipeline = PricePredictionPipeline()

        if not request.address or len(request.address.strip()) < 5:
            raise HTTPException(
                status_code=400, detail="Address must be at least 5 characters")

        logger.info(f"Processing prediction for: {request.address}")
        result = pipeline.predict_price(request.address)

        return PriceResponse(
            address=result['address'],
            predicted_price=result['predicted_price'],
            confidence=result['confidence'],
            error_margin=result['error_margin'],
            error_margin_low=result['error_margin_low'],
            error_margin_high=result['error_margin_high'],
            all_16_features=result['all_16_features'],
            school_district=result['school_district'],
            school_rating=result['school_rating'],
            timestamp=result['timestamp']
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(addresses: list[AddressRequest]):
    """
    Predict prices for multiple addresses.

    Args:
        addresses: List of address requests

    Returns:
        List of predictions
    """
    results = []
    for req in addresses:
        try:
            result = await predict_price(req)
            results.append(result)
        except HTTPException as e:
            results.append({
                "address": req.address,
                "error": e.detail
            })
    return results


@app.get("/v1/meta/capabilities")
async def get_capabilities():
    """
    Return API capabilities and model metadata.
    Used by training pipeline to understand API features.
    """
    return {
        "contract_version": "2.0.0",
        "model_name": "House Price Predictor",
        "model_version": "2.0.0",
        "feature_policy_name": "default",
        "model_expected_features": [
            "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Fireplaces", "GarageCars", "GarageArea",
            "NeighborhoodScore", "CensusMedianValue", "MedianIncomeK", "OwnerOccupiedRate"
        ]
    }


@app.get("/v1/meta/live-feature-candidates")
async def get_live_feature_candidates(
    limit: int = 100,
    offset: int = 0,
    min_completeness_score: float = 0.8,
    include_reused: bool = False
):
    """
    Return live feature candidates from the training dataset.
    This endpoint is used by the training pipeline to fetch data.

    For demo purposes, loads from data/processed/final_training_dataset.csv
    In production, this would fetch from a database of prediction audit logs.
    """
    try:
        csv_path = Path(__file__).parent.parent.parent / "data" / \
            "processed" / "final_training_dataset.csv"
        if not csv_path.exists():
            return {
                "items": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "message": "Training dataset not found"
            }

        # Load the training dataset
        df = pd.read_csv(csv_path)

        # Extract features and format as candidates
        items = []
        for idx, row in df.iloc[offset:offset+limit].iterrows():
            # Extract numeric features from CSV columns
            features = {
                "LotArea": float(row.get("LOT SIZE", 5000)) if pd.notna(row.get("LOT SIZE")) else 5000,
                "OverallQual": float(row.get("OVERALL QUALITY", 7)) if pd.notna(row.get("OVERALL QUALITY")) else 7,
                "OverallCond": float(row.get("OVERALL CONDITION", 7)) if pd.notna(row.get("OVERALL CONDITION")) else 7,
                "YearBuilt": float(row.get("YEAR BUILT", 2000)) if pd.notna(row.get("YEAR BUILT")) else 2000,
                "YearRemodAdd": float(row.get("YEAR BUILT", 2000)) if pd.notna(row.get("YEAR BUILT")) else 2000,
                "GrLivArea": float(row.get("SQUARE FEET", 2000)) if pd.notna(row.get("SQUARE FEET")) else 2000,
                "FullBath": float(row.get("BATHS", 2)) if pd.notna(row.get("BATHS")) else 2,
                "HalfBath": 0,
                "BedroomAbvGr": float(row.get("BEDS", 3)) if pd.notna(row.get("BEDS")) else 3,
                "TotRmsAbvGrd": float(row.get("BEDS", 6)) if pd.notna(row.get("BEDS")) else 6,
                "Fireplaces": 0,
                "GarageCars": 2,
                "GarageArea": 400,
                "City": str(row.get("CITY", "Unknown")) if pd.notna(row.get("CITY")) else "Unknown",
                "ZipCode": str(row.get("ZIP OR POSTAL CODE", "00000")) if pd.notna(row.get("ZIP OR POSTAL CODE")) else "00000",
                "State": str(row.get("STATE OR PROVINCE", "NA")) if pd.notna(row.get("STATE OR PROVINCE")) else "NA",
                "SchoolDistrictRating": 6.5,
                "WalkScore": float(50 + (idx % 50)),
                "HOAFee": float(round((idx % 10) * 50)),
                "PricePerSqft": round(float(row.get("PRICE", 300000)) / max(float(row.get("SQUARE FEET", 1500)), 1), 2) if pd.notna(row.get("PRICE")) else 180.0,
                "LandValue": round(float(row.get("PRICE", 300000)) * 0.25, 2) if pd.notna(row.get("PRICE")) else 75000.0,
                "NeighborhoodScore": 50 + (idx % 50),
                "CensusMedianValue": float(row.get("PRICE", 250000)) if pd.notna(row.get("PRICE")) else 250000,
                "MedianIncomeK": 75,
                "OwnerOccupiedRate": 0.75
            }

            item = {
                "predicted_price": float(row.get("PRICE", 300000)) if pd.notna(row.get("PRICE")) else 300000,
                "features": features,
                "normalized_address": {
                    "latitude": 33.7490 + (idx % 100) * 0.001,
                    "longitude": -84.3880 + (idx % 100) * 0.001
                }
            }
            items.append(item)

        return {
            "items": items,
            "total": len(df),
            "limit": limit,
            "offset": offset,
            "message": f"Loaded {len(items)} candidates from training dataset"
        }
    except Exception as e:
        logger.error(f"Error loading candidates: {e}")
        return {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
