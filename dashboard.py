import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# Set page config
st.set_page_config(
    page_title="Housing Price Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Keyframe animations */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Main page background */
    .stMain {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        animation: slideInDown 0.6s ease-out;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards with shadow and gradient */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        animation: slideInDown 0.5s ease-out backwards;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3);
        animation: pulse 0.6s ease-in-out;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600;
        color: #555;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        font-weight: 700;
        color: #667eea;
        animation: slideInDown 0.6s ease-out;
    }
    
    /* Section headings */
    h1 {
        color: #667eea !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.05);
        animation: slideInDown 0.6s ease-out;
    }
    
    h2 {
        color: #764ba2 !important;
        font-weight: 700 !important;
        animation: slideInDown 0.6s ease-out;
    }
    
    h3 {
        margin: 0.5rem 0 !important;
        font-size: 1.1rem !important;
        color: #555 !important;
        font-weight: 600;
        animation: slideInDown 0.5s ease-out;
    }
    
    h4 {
        margin: 0.2rem 0 !important;
        font-size: 0.95rem !important;
        color: #667eea !important;
    }
    
    /* Containers with nice borders */
    [data-testid="stContainer"] {
        border-radius: 12px;
        animation: fadeIn 0.7s ease-in;
    }
    
    /* Form inputs styling */
    input, textarea, select {
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        transition: all 0.3s ease;
    }
    
    input:focus, textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Buttons styling */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid !important;
    }
    
    /* Markdown text */
    p {
        color: #333 !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data using the trained model's data pipeline
@st.cache_data
def load_data():
    import os
    import sys
    from pathlib import Path
    from src.house_price_prediction.data import load_dataset
    from src.house_price_prediction.config import load_settings
    
    # Load using the same settings as training
    settings = load_settings()
    df = load_dataset(settings.raw_data_path)
    return df

# Load trained model from pickle/joblib file
@st.cache_resource
def load_trained_model():
    """Load the most recently modified model artifact (.pkl or .joblib) from the models directory."""
    from pathlib import Path
    from src.house_price_prediction.model import load_model_artifact

    models_dir = Path(__file__).parent / "models"
    candidates = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))

    if not candidates:
        st.warning(f"⚠️ No model files found in {models_dir}")
        return None

    model_path = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        artifact = load_model_artifact(model_path)
        st.sidebar.info(f"Model loaded: `{model_path.name}`")
        return artifact
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df = load_data()
model_artifact = load_trained_model()


def call_api(method: str, base_url: str, path: str, payload: dict | None = None, params: dict | None = None):
    """Execute a backend API request and normalize response handling for UI rendering."""
    target_url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method,
            url=target_url,
            json=payload,
            params=params,
            timeout=20,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        return response.status_code, body, target_url
    except requests.RequestException as exc:
        return None, {"error": str(exc)}, target_url


def calculate_market_metrics(dataset: pd.DataFrame) -> dict:
    """Calculate 5 key metrics from the housing dataset."""
    if dataset.empty:
        return {}
    
    metrics = {}
    
    # 1. Neighborhood Stats
    metrics["neighborhood_stats"] = {
        "avg_price": dataset["price"].mean(),
        "median_price": dataset["price"].median(),
        "price_range": (dataset["price"].min(), dataset["price"].max()),
        "total_properties": len(dataset),
    }
    
    # 2. Property Characteristics
    metrics["property_characteristics"] = {
        "avg_lot_area": dataset["sqft_lot"].mean(),
        "avg_grade": dataset["grade"].mean(),
        "avg_condition": dataset["condition"].mean(),
        "avg_living_area": dataset["sqft_living"].mean(),
        "avg_bathrooms": dataset["bathrooms"].mean(),
        "avg_bedrooms": dataset["bedrooms"].mean(),
        "avg_floors": dataset["floors"].mean(),
    }
    
    # 3. Market Data (Price per sqft)
    dataset_copy = dataset.copy()
    dataset_copy["price_per_sqft"] = dataset_copy["price"] / dataset_copy["sqft_living"]
    dataset_copy = dataset_copy[dataset_copy["price_per_sqft"] > 0]
    metrics["market_data"] = {
        "avg_price_per_sqft": dataset_copy["price_per_sqft"].mean(),
        "median_price_per_sqft": dataset_copy["price_per_sqft"].median(),
    }
    
    # 4. Age Estimate (Average Year Built and Age)
    metrics["age_estimate"] = {
        "avg_year_built": dataset["yr_built"].mean(),
        "avg_age": (2026 - dataset["yr_built"].mean()),
        "oldest_year": dataset["yr_built"].min(),
        "newest_year": dataset["yr_built"].max(),
    }
    
    # 5. Tax Estimate (Approximate based on average price)
    # Typical property tax rate ~1.2% of property value per year in US
    tax_rate = 0.012
    metrics["tax_estimate"] = {
        "avg_annual_tax": (dataset["price"].mean() * tax_rate),
        "annual_tax_rate": f"{tax_rate * 100:.1f}%",
    }
    
    return metrics


def lookup_state_key(slot_index: int, name: str) -> str:
    return f"lookup_{slot_index}_{name}"


def render_prediction_result(slot_index: int) -> None:
    prediction = st.session_state.get(lookup_state_key(slot_index, "prediction"))
    prediction_error = st.session_state.get(lookup_state_key(slot_index, "prediction_error"))

    if prediction:
        st.success("✅ Prediction Complete!")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            price = prediction.get("predicted_price")
            if price is not None:
                st.metric("Predicted Price", f"${price:,.0f}")

        with res_col2:
            completeness = prediction.get("feature_snapshot", {}).get("completeness_score")
            if completeness is not None:
                st.metric("Data Completeness", f"{completeness:.1%}")

        with res_col3:
            request_id = str(prediction.get("request_id", "N/A"))
            st.metric("Request ID", request_id[:8] + "..." if len(request_id) > 8 else request_id)

        key_features = prediction.get("feature_snapshot", {}).get("features", {})
        if key_features:
            st.markdown("**Key Property Features:**")
            feat_cols = st.columns(min(len(key_features), 4))
            for idx, (feat_name, feat_val) in enumerate(list(key_features.items())[:4]):
                with feat_cols[idx % 4]:
                    st.metric(feat_name, feat_val)
    elif prediction_error:
        st.error(prediction_error.get("message", "Prediction failed"))
        detail = prediction_error.get("detail")
        if detail:
            if isinstance(detail, str):
                st.error(f"Error: {detail}")
            else:
                st.json(detail)


def render_market_metrics(metrics: dict) -> None:
    """Display the 5 calculated market metrics."""
    if not metrics:
        return

    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
        <h2 style="color: white; margin: 0; font-size: 1.6em; letter-spacing: 0.5px;">🏠 Market Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Neighborhood Stats — 3 + 2 layout so values never truncate ─────────
    if "neighborhood_stats" in metrics:
        st.markdown("#### 📍 Neighborhood Statistics")
        ns = metrics["neighborhood_stats"]
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Avg Price",    f"${ns['avg_price']:,.0f}")
        r1c2.metric("Median Price", f"${ns['median_price']:,.0f}")
        r1c3.metric("# Properties", f"{ns['total_properties']:,}")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Min Price", f"${ns['price_range'][0]:,.0f}")
        r2c2.metric("Max Price", f"${ns['price_range'][1]:,.0f}")

    # ── 2. Property Characteristics — 4-across grid, one metric per cell ──────
    if "property_characteristics" in metrics:
        st.markdown("#### 🏡 Average Property Characteristics")
        pc = metrics["property_characteristics"]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Living Area",  f"{pc['avg_living_area']:,.0f} sqft")
        rc2.metric("Lot Area",     f"{pc['avg_lot_area']:,.0f} sqft")
        rc3.metric("Bedrooms",     f"{pc['avg_bedrooms']:.1f}")
        rc4.metric("Bathrooms",    f"{pc['avg_bathrooms']:.1f}")

        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("Grade (1-13)",    f"{pc['avg_grade']:.1f}")
        rc6.metric("Condition (1-5)", f"{pc['avg_condition']:.1f}")
        rc7.metric("Floors",          f"{pc['avg_floors']:.1f}")

    # ── 3 & 5. Market Data + Tax side by side ─────────────────────────────────
    if "market_data" in metrics or "tax_estimate" in metrics:
        st.markdown("#### 💰 Market & Tax Data")
        md_col1, md_col2, md_col3, md_col4 = st.columns(4)
        if "market_data" in metrics:
            md = metrics["market_data"]
            md_col1.metric("Avg $/sqft",    f"${md['avg_price_per_sqft']:.2f}")
            md_col2.metric("Median $/sqft", f"${md['median_price_per_sqft']:.2f}")
        if "tax_estimate" in metrics:
            te = metrics["tax_estimate"]
            md_col3.metric("Est. Annual Tax", f"${te['avg_annual_tax']:,.0f}")
            md_col4.metric("Tax Rate",        te["annual_tax_rate"])

    # ── 4. Age Estimate ────────────────────────────────────────────────────────
    if "age_estimate" in metrics:
        st.markdown("#### 🗓️ Property Age")
        ae = metrics["age_estimate"]
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Avg Year Built", f"{ae['avg_year_built']:.0f}")
        ac2.metric("Avg Age",        f"{ae['avg_age']:.0f} yrs")
        ac3.metric("Oldest Built",   f"{ae['oldest_year']:.0f}")
        ac4.metric("Newest Built",   f"{ae['newest_year']:.0f}")


import re as _re

# ── Sensible median defaults for the local prediction form ───────────────────
_LOCAL_DEFAULTS = {
    "LotArea": 8000,
    "OverallQual": 6,
    "OverallCond": 5,
    "YearBuilt": 1990,
    "YearRemodAdd": 2000,
    "GrLivArea": 1500,
    "FullBath": 2,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 7,
    "Fireplaces": 0,
    "GarageCars": 2,
    "GarageArea": 400,
    "NeighborhoodScore": 5.0,
    "CensusMedianValue": 200000,
    "MedianIncomeK": 55.0,
    "OwnerOccupiedRate": 0.65,
    "SchoolDistrictRating": 6.0,
    "WalkScore": 50,
    "HOAFee": 0,
    "PricePerSqft": 150,
    "LandValue": 50000,
}


def _render_local_prediction_form(slot_index: int, pipeline, parsed_addr: dict) -> None:
    """Render a property-feature form and predict using the locally loaded model (no API needed)."""
    st.markdown("---")
    st.info(
        "💡 **API server is offline.** Use the form below to get an instant price estimate "
        "directly from the locally loaded model — no backend required."
    )
    st.subheader("🔮 Local Prediction")

    with st.form(f"local_predict_form_{slot_index}"):
        st.markdown("**📍 Location** (pre-filled from your address)")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            l_city = st.text_input("City", value=parsed_addr.get("city", ""))
        with lc2:
            l_state = st.text_input("State (2-letter)", value=parsed_addr.get("state", ""), max_chars=2)
        with lc3:
            l_zip = st.text_input("ZIP Code", value=parsed_addr.get("postal", ""), max_chars=5)

        st.markdown("**🏠 Property Details**")
        pd1, pd2, pd3 = st.columns(3)
        with pd1:
            l_proptype = st.selectbox("Property Type", ["single_family", "townhouse", "luxury"])
        with pd2:
            l_yr_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=_LOCAL_DEFAULTS["YearBuilt"])
        with pd3:
            l_yr_remod = st.number_input("Year Last Remodeled", min_value=1800, max_value=2026, value=_LOCAL_DEFAULTS["YearRemodAdd"])

        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            l_qual = st.slider("Overall Quality (1–10)", 1, 10, _LOCAL_DEFAULTS["OverallQual"])
        with qa2:
            l_cond = st.slider("Overall Condition (1–10)", 1, 10, _LOCAL_DEFAULTS["OverallCond"])
        with qa3:
            l_gr_liv = st.number_input("Living Area (sqft)", min_value=100, max_value=20000, value=_LOCAL_DEFAULTS["GrLivArea"])
        with qa4:
            l_lot = st.number_input("Lot Area (sqft)", min_value=100, max_value=1000000, value=_LOCAL_DEFAULTS["LotArea"])

        rm1, rm2, rm3, rm4 = st.columns(4)
        with rm1:
            l_beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=_LOCAL_DEFAULTS["BedroomAbvGr"])
        with rm2:
            l_fbath = st.number_input("Full Baths", min_value=0, max_value=10, value=_LOCAL_DEFAULTS["FullBath"])
        with rm3:
            l_hbath = st.number_input("Half Baths", min_value=0, max_value=10, value=_LOCAL_DEFAULTS["HalfBath"])
        with rm4:
            l_rooms = st.number_input("Total Rooms", min_value=1, max_value=30, value=_LOCAL_DEFAULTS["TotRmsAbvGrd"])

        gr1, gr2, gr3 = st.columns(3)
        with gr1:
            l_fire = st.number_input("Fireplaces", min_value=0, max_value=5, value=_LOCAL_DEFAULTS["Fireplaces"])
        with gr2:
            l_garage_cars = st.number_input("Garage (# cars)", min_value=0, max_value=5, value=_LOCAL_DEFAULTS["GarageCars"])
        with gr3:
            l_garage_area = st.number_input("Garage Area (sqft)", min_value=0, max_value=5000, value=_LOCAL_DEFAULTS["GarageArea"])

        with st.expander("🌍 Neighborhood & Economic Factors (optional — defaults are median values)"):
            ne1, ne2, ne3 = st.columns(3)
            with ne1:
                l_neighborhood = st.slider("Neighborhood Score", 0.0, 10.0, float(_LOCAL_DEFAULTS["NeighborhoodScore"]), 0.1)
                l_school = st.slider("School District Rating", 0.0, 10.0, float(_LOCAL_DEFAULTS["SchoolDistrictRating"]), 0.1)
            with ne2:
                l_income = st.number_input("Median Income ($K)", min_value=0.0, max_value=500.0, value=float(_LOCAL_DEFAULTS["MedianIncomeK"]))
                l_walkscore = st.slider("Walk Score (0–100)", 0, 100, _LOCAL_DEFAULTS["WalkScore"])
            with ne3:
                l_census_val = st.number_input("Census Median Value ($)", min_value=0, max_value=2000000, value=_LOCAL_DEFAULTS["CensusMedianValue"])
                l_hoa = st.number_input("HOA Fee ($/mo)", min_value=0, max_value=5000, value=_LOCAL_DEFAULTS["HOAFee"])
            en1, en2 = st.columns(2)
            with en1:
                l_owner_rate = st.slider("Owner Occupied Rate", 0.0, 1.0, float(_LOCAL_DEFAULTS["OwnerOccupiedRate"]), 0.01)
                l_land_val = st.number_input("Land Value ($)", min_value=0, max_value=2000000, value=_LOCAL_DEFAULTS["LandValue"])
            with en2:
                l_price_sqft = st.number_input("Price Per Sqft estimate ($)", min_value=0, max_value=5000, value=_LOCAL_DEFAULTS["PricePerSqft"])

        local_submit = st.form_submit_button("🔮 Predict Locally", use_container_width=True)

    local_pred_key = f"local_prediction_{slot_index}"
    if local_submit:
        feature_row = {
            "LotArea": int(l_lot),
            "OverallQual": int(l_qual),
            "OverallCond": int(l_cond),
            "YearBuilt": int(l_yr_built),
            "YearRemodAdd": int(l_yr_remod),
            "GrLivArea": int(l_gr_liv),
            "FullBath": int(l_fbath),
            "HalfBath": int(l_hbath),
            "BedroomAbvGr": int(l_beds),
            "TotRmsAbvGrd": int(l_rooms),
            "Fireplaces": int(l_fire),
            "GarageCars": int(l_garage_cars),
            "GarageArea": int(l_garage_area),
            "NeighborhoodScore": float(l_neighborhood),
            "CensusMedianValue": int(l_census_val),
            "MedianIncomeK": float(l_income),
            "OwnerOccupiedRate": float(l_owner_rate),
            "PropertyType": l_proptype,
            "City": l_city.strip(),
            "ZipCode": l_zip.strip(),
            "State": l_state.strip().upper(),
            "SchoolDistrictRating": float(l_school),
            "WalkScore": int(l_walkscore),
            "HOAFee": int(l_hoa),
            "PricePerSqft": int(l_price_sqft),
            "LandValue": int(l_land_val),
        }
        try:
            row_df = pd.DataFrame([feature_row])
            predicted_price = float(pipeline.predict(row_df)[0])
            st.session_state[local_pred_key] = predicted_price
        except Exception as exc:
            st.session_state[local_pred_key] = None
            st.error(f"❌ Local prediction failed: {exc}")

    local_pred = st.session_state.get(local_pred_key)
    if local_pred is not None:
        st.success("### 🏠 Price Estimate Ready")
        lpr1, lpr2, lpr3 = st.columns(3)
        lpr1.metric("Predicted Price", f"${local_pred:,.0f}")
        lpr2.metric("Model", "LightGBM (local)")
        lpr3.metric("Source", "Offline / No API")


def _parse_address_client(full: str) -> dict:
    """Client-side parser: extract street, city, state, zip from a free-form US address."""
    s = full.strip()

    # Pattern 1: ..., City, ST 00000  or  ..., City ST 00000
    m = _re.search(r",?\s*([^,]+?),?\s*\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\s*$", s, _re.IGNORECASE)
    if m:
        city   = m.group(1).strip().strip(',')
        state  = m.group(2).upper()
        zipcode = m.group(3)[:5]
        street = s[:m.start()].strip().strip(',')
        return {"line1": street, "city": city, "state": state, "postal": zipcode}

    # Pattern 2: trailing ST 00000 (space-separated)
    m2 = _re.search(r'\s+([A-Z]{2})\s+(\d{5})\s*$', s, _re.IGNORECASE)
    if m2:
        state   = m2.group(1).upper()
        zipcode = m2.group(2)
        before  = s[:m2.start()].strip()
        if ',' in before:
            parts = [p.strip() for p in before.rsplit(',', 1)]
            street, city = parts[0], parts[1]
        else:
            toks = before.split()
            city   = toks[-1] if toks else ""
            street = " ".join(toks[:-1])
        return {"line1": street, "city": city, "state": state, "postal": zipcode}

    # Pattern 3: City, ST only
    m3 = _re.search(r",?\s*([^,]+?),?\s*\b([A-Z]{2})\s*$", s, _re.IGNORECASE)
    if m3:
        city   = m3.group(1).strip().strip(',')
        state  = m3.group(2).upper()
        street = s[:m3.start()].strip().strip(',')
        return {"line1": street, "city": city, "state": state, "postal": ""}

    # Fallback: pass raw string through unchanged
    return {"line1": s, "city": "", "state": "", "postal": ""}


def render_lookup_slot(slot_index: int, api_base_url: str) -> dict | None:
    normalized_key = lookup_state_key(slot_index, "normalized")
    prediction_key = lookup_state_key(slot_index, "prediction")
    prediction_error_key = lookup_state_key(slot_index, "prediction_error")

    if normalized_key not in st.session_state:
        st.session_state[normalized_key] = None
    if prediction_key not in st.session_state:
        st.session_state[prediction_key] = None
    if prediction_error_key not in st.session_state:
        st.session_state[prediction_error_key] = None

    slot_label = f"Search Address {slot_index + 1}"
    with st.container(border=True):
        st.markdown(f"### {slot_label}")

        with st.form(f"address_lookup_form_{slot_index}"):
            # ── Primary: single free-form address field ───────────────────
            st.text_input(
                "📍 Full Address",
                placeholder="e.g.  123 Peachtree St NE, Atlanta, GA 30308   or   500 NW 1st Ave Miami FL 33128",
                help=(
                    "Type the complete address in any of these formats:\n"
                    "- 123 Main St, City, ST 00000\n"
                    "- 123 Main St, City ST 00000\n"
                    "- 123 Main St City State ZipCode\n\n"
                    "Use the fields below only if you need to override individual parts."
                ),
                key=lookup_state_key(slot_index, "full_addr"),
            )

            # ── Secondary: manual structured override ─────────────────────
            with st.expander("🔧 Override individual fields (optional)"):
                col1, col2 = st.columns([2, 1], gap="medium")
                with col1:
                    st.text_input(
                        "Street / Address Line 1",
                        placeholder="123 Main Street",
                        key=lookup_state_key(slot_index, "line1"),
                    )
                with col2:
                    st.text_input(
                        "Unit / Apt",
                        placeholder="Apt 4B",
                        key=lookup_state_key(slot_index, "line2"),
                    )
                col3, col4, col5, col6 = st.columns([2, 1, 1, 1], gap="small")
                with col3:
                    st.text_input(
                        "City",
                        placeholder="Atlanta",
                        key=lookup_state_key(slot_index, "city"),
                    )
                with col4:
                    st.text_input(
                        "State",
                        placeholder="GA",
                        max_chars=2,
                        key=lookup_state_key(slot_index, "state"),
                    )
                with col5:
                    st.text_input(
                        "ZIP Code",
                        placeholder="30308",
                        max_chars=10,
                        key=lookup_state_key(slot_index, "postal"),
                    )
                with col6:
                    st.text_input(
                        "Country",
                        max_chars=2,
                        key=lookup_state_key(slot_index, "country"),
                        value=st.session_state.get(lookup_state_key(slot_index, "country"), "US"),
                    )

            search_submitted = st.form_submit_button("🔍 Search Address", use_container_width=True)

        if search_submitted:
            full_addr_raw  = st.session_state.get(lookup_state_key(slot_index, "full_addr"), "").strip()
            manual_line1   = st.session_state.get(lookup_state_key(slot_index, "line1"), "").strip()
            manual_city    = st.session_state.get(lookup_state_key(slot_index, "city"), "").strip()
            manual_state   = st.session_state.get(lookup_state_key(slot_index, "state"), "").strip().upper()
            manual_line2   = st.session_state.get(lookup_state_key(slot_index, "line2"), "").strip()
            manual_postal  = st.session_state.get(lookup_state_key(slot_index, "postal"), "").strip()
            manual_country = st.session_state.get(lookup_state_key(slot_index, "country"), "US").strip()

            if not full_addr_raw and not manual_line1:
                st.warning("⚠️ Enter an address above to search.")
            else:
                # If free-form provided, parse it to fill any missing manual fields
                if full_addr_raw:
                    parsed = _parse_address_client(full_addr_raw)
                    line1   = manual_line1   or parsed.get("line1", "")
                    city    = manual_city    or parsed.get("city", "")
                    state   = manual_state   or parsed.get("state", "")
                    postal  = manual_postal  or parsed.get("postal", "")
                else:
                    line1, city, state, postal = manual_line1, manual_city, manual_state, manual_postal

                # Build payload — always send full_address for maximum geocoding accuracy
                address_parts = [p for p in [line1, manual_line2, city, f"{state} {postal}".strip()] if p]
                canonical_full = full_addr_raw or ", ".join(address_parts)

                lookup_payload = {"full_address": canonical_full, "country": manual_country or "US"}
                # Also send structured fields when available (backend uses whichever is more complete)
                if line1:   lookup_payload["address_line_1"] = line1
                if city:    lookup_payload["city"]           = city
                if state:   lookup_payload["state"]          = state
                if postal:  lookup_payload["postal_code"]    = postal
                if manual_line2: lookup_payload["address_line_2"] = manual_line2

                with st.spinner(f"🌍 Looking up address for {slot_label.lower()}..."):
                    sc, body, _url = call_api("POST", api_base_url, "/v1/properties/normalize", payload=lookup_payload)

                if sc == 200 and isinstance(body, dict):
                    st.session_state[normalized_key] = body
                    st.session_state[prediction_key] = None
                    st.session_state[prediction_error_key] = None
                    # Clear any stale local-fallback state when API succeeds
                    st.session_state.pop(f"local_fallback_{slot_index}", None)
                    st.session_state.pop(f"local_prediction_{slot_index}", None)
                    st.success("✅ Address found!")
                else:
                    st.session_state[normalized_key] = None
                    # Persist parsed address so the local prediction form stays visible on rerun
                    st.session_state[f"local_fallback_{slot_index}"] = {
                        "city": city, "state": state, "postal": postal
                    }
                    error_detail = ""
                    if sc is None:
                        error_detail = "API server is not reachable — is the backend running?"
                    elif isinstance(body, dict):
                        detail = body.get("detail", "")
                        if isinstance(detail, list):
                            error_detail = "; ".join(d.get("msg", str(d)) for d in detail)
                        elif detail:
                            error_detail = str(detail)
                    st.error(f"❌ Address lookup failed (HTTP {sc}){': ' + error_detail if error_detail else ''}")
                    if sc is not None:
                        with st.expander("Response detail"):
                            st.json(body)

        normalized = st.session_state.get(normalized_key)
        if not normalized:
            local_fallback = st.session_state.get(f"local_fallback_{slot_index}")
            if local_fallback and model_artifact is not None:
                _render_local_prediction_form(slot_index, model_artifact.model, local_fallback)
            elif local_fallback and model_artifact is None:
                st.warning("⚠️ No trained model found. Train a model using `scripts/train.py` to enable local predictions.")
            else:
                st.info("Enter an address above and click Search Address.")
            return None

        col_addr, col_coords = st.columns([2, 1])

        with col_addr:
            st.subheader("📬 Normalized Address")
            addr_display = f"{normalized.get('address_line_1', '')}"
            if normalized.get("address_line_2"):
                addr_display += f", {normalized.get('address_line_2')}"
            addr_display += (
                f"\n{normalized.get('city', '')}, {normalized.get('state', '')} "
                f"{normalized.get('postal_code', '')}"
            )
            if normalized.get("country"):
                addr_display += f"\n{normalized.get('country', '')}"
            st.info(addr_display)

            if normalized.get("formatted_address"):
                st.caption(f"Formatted: {normalized['formatted_address']}")

        with col_coords:
            latitude = normalized.get("latitude")
            longitude = normalized.get("longitude")
            if latitude is not None and longitude is not None:
                st.metric("Latitude", f"{latitude:.4f}")
                st.metric("Longitude", f"{longitude:.4f}")
                if normalized.get("geocoding_source"):
                    st.caption(f"📡 Source: {normalized['geocoding_source']}")

        # Map pin
        latitude = normalized.get("latitude")
        longitude = normalized.get("longitude")
        if latitude is not None and longitude is not None:
            st.map(
                pd.DataFrame([{"lat": latitude, "lon": longitude}]),
                latitude="lat",
                longitude="lon",
                zoom=13,
            )

        # ── Predict Price ─────────────────────────────────────────────────
        st.markdown("---")

        # Show existing prediction result prominently BEFORE market context
        existing_prediction = st.session_state.get(prediction_key)
        if existing_prediction:
            price = existing_prediction.get("predicted_price")
            if price is not None:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 28px 32px; border-radius: 14px; margin: 12px 0 20px 0;
                                text-align: center; box-shadow: 0 8px 30px rgba(102,126,234,0.4);">
                        <p style="color: rgba(255,255,255,0.85); margin: 0 0 6px 0; font-size: 1rem; font-weight: 500;">
                            Estimated House Price
                        </p>
                        <p style="color: white; margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: 1px;">
                            ${price:,.0f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cs_col, rid_col = st.columns(2)
                completeness = existing_prediction.get("feature_snapshot", {}).get("completeness_score")
                if completeness is not None:
                    cs_col.metric("Data Completeness", f"{completeness:.1%}")
                request_id = str(existing_prediction.get("request_id", "N/A"))
                rid_col.metric("Request ID", request_id[:8] + "…" if len(request_id) > 8 else request_id)

        prediction_error = st.session_state.get(prediction_error_key)
        if prediction_error:
            st.error(prediction_error["message"])
            detail = prediction_error.get("detail")
            if detail:
                st.error(f"Detail: {detail}" if isinstance(detail, str) else "")
                if not isinstance(detail, str):
                    st.json(detail)

        st.subheader("🔮 Get Price Prediction")
        ep_col, btn_col = st.columns([3, 1])
        with ep_col:
            st.text_input(
                "Your Email (optional)",
                placeholder="you@example.com",
                key=lookup_state_key(slot_index, "requested_by"),
                label_visibility="collapsed",
            )

        if st.button("Predict House Price", use_container_width=True, key=lookup_state_key(slot_index, "predict"), type="primary"):
            formatted = normalized.get("formatted_address") or normalized.get("address_line_1", "")
            pred_payload = {
                "full_address": formatted,
                "address_line_1": normalized.get("address_line_1"),
                "city": normalized.get("city"),
                "state": normalized.get("state"),
                "postal_code": normalized.get("postal_code"),
                "country": normalized.get("country"),
            }
            if normalized.get("address_line_2"):
                pred_payload["address_line_2"] = normalized.get("address_line_2")

            requested_by = st.session_state.get(lookup_state_key(slot_index, "requested_by"), "")
            if requested_by.strip():
                pred_payload["requested_by"] = requested_by.strip()

            with st.spinner(f"🧠 Analyzing property for {slot_label.lower()}..."):
                pred_sc, pred_body, _pred_url = call_api(
                    "POST",
                    api_base_url,
                    "/v1/predictions",
                    payload=pred_payload,
                )

            if pred_sc == 201 and isinstance(pred_body, dict):
                st.session_state[prediction_key] = pred_body
                st.session_state[prediction_error_key] = None
                st.session_state["last_prediction_id"] = str(pred_body.get("prediction_id", ""))
                st.rerun()
            else:
                st.session_state[prediction_key] = None
                st.session_state[prediction_error_key] = {
                    "message": f"❌ Prediction failed (Status {pred_sc})",
                    "detail": pred_body.get("detail") if isinstance(pred_body, dict) else pred_body,
                }
                st.rerun()

        # ── Market context (shown below the prediction) ────────────────────
        with st.expander("📊 Dataset Market Context", expanded=False):
            metrics = calculate_market_metrics(df)
            render_market_metrics(metrics)
        return normalized

# Sidebar
st.sidebar.title("🏠 Housing Dashboard")
st.sidebar.markdown("---")

# Get API URL from environment or default to localhost
import os
default_api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

api_base_url = st.sidebar.text_input(
    "Live API Base URL",
    value=st.session_state.get("api_base_url", default_api_url),
    help="Use the backend URL you want the UI to test against.",
)
st.session_state["api_base_url"] = api_base_url

# Show environment info
if "onrender.com" in api_base_url:
    st.sidebar.success("🚀 Connected to Render deployment")
elif "localhost" in api_base_url or "127.0.0.1" in api_base_url:
    st.sidebar.info("🏠 Connected to local development server")
else:
    st.sidebar.info(f"📡 Custom API: {api_base_url}")

if st.sidebar.button("Check API Runtime", use_container_width=True):
    sc, body, url = call_api("GET", api_base_url, "/v1/health")
    st.session_state["api_runtime_health"] = {
        "status_code": sc,
        "body": body,
        "url": url,
    }

runtime_health = st.session_state.get("api_runtime_health")
if runtime_health is not None:
    status_code = runtime_health.get("status_code")
    health_body = runtime_health.get("body")
    if status_code == 200 and isinstance(health_body, dict):
        st.sidebar.success("API reachable")
        st.sidebar.caption(
            f"Providers: {health_body.get('geocoding_provider', 'unknown')} / "
            f"{health_body.get('property_data_provider', 'unknown')}"
        )
        st.sidebar.caption(
            f"Mock predictor: {health_body.get('mock_predictor_enabled', 'unknown')}"
        )
        geocoding_provider = str(health_body.get("geocoding_provider", "")).strip().lower()
        property_provider = str(health_body.get("property_data_provider", "")).strip().lower()
        mock_enabled = bool(health_body.get("mock_predictor_enabled", False))

        if mock_enabled:
            st.sidebar.warning(
                "Predictions are currently using mock mode. Set ENABLE_MOCK_PREDICTOR=false "
                "for live ML model inference."
            )
        if geocoding_provider == "fake" or property_provider == "fake":
            st.sidebar.warning(
                "At least one provider is fake. Use free-fallback providers for live address enrichment."
            )
    else:
        st.sidebar.error(f"Health check failed ({status_code}): {health_body}")

# Page selection
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Analysis", "Feature Exploration", "Statistics", "Address Lookup", "Live API Tester"]
)

# ==================== PAGE: OVERVIEW ====================
if page == "Overview":
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🏠 Housing Price Prediction & Economic Indicators</h1>
        <p style="color: #666; font-size: 1.1rem;">Explore the housing dataset with interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status
    if model_artifact:
        st.success("✅ **Trained Model Loaded** from `models/house_price_model.pkl`")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("Model Name", model_artifact.metadata.model_name or "Random Forest")
        with col_meta2:
            st.metric("Features", len(model_artifact.metadata.feature_columns))
        with col_meta3:
            st.metric("Version", model_artifact.metadata.model_version or "1.0")
    else:
        st.warning("⚠️ No trained model found. Train a model using `scripts/train.py`")
    
    st.markdown("---")
    
    # Key metrics — 5 cards so each price value has room to breathe
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Avg Price",        f"${df['price'].mean():,.0f}")
    col3.metric("Min Price",        f"${df['price'].min():,.0f}")
    col4.metric("Max Price",        f"${df['price'].max():,.0f}")
    col5.metric("Avg Bedrooms",     f"{df['bedrooms'].mean():.1f}")
    
    st.markdown("---")
    
    # Price distribution
    st.subheader("📊 Price Distribution")
    
    fig = px.histogram(
        df, 
        x='price', 
        nbins=50,
        labels={'price': 'Price ($)'},
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.add_annotation(
        text="<b>Price Distribution</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=16, color="#667eea"),
        xanchor="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    fig = px.box(
        df, 
        y='price',
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.add_annotation(
        text="<b>Price Box Plot</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=16, color="#667eea"),
        xanchor="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==================== PAGE: DATA ANALYSIS ====================
elif page == "Data Analysis":
    st.title("📈 Data Analysis")
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = st.slider(
            "Min Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].min()),
            step=10000
        )
    
    with col2:
        max_price = st.slider(
            "Max Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].max()),
            step=10000
        )
    
    with col3:
        bedrooms = st.multiselect(
            "Bedrooms",
            sorted(df['bedrooms'].unique()),
            default=sorted(df['bedrooms'].unique())
        )
    
    # Apply filters
    filtered_df = df[
        (df['price'] >= min_price) & 
        (df['price'] <= max_price) &
        (df['bedrooms'].isin(bedrooms))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} properties** (out of {len(df)})")
    
    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("🔗 Correlation with Price")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()['price'].sort_values(ascending=False)
    correlation_df = correlation.reset_index()
    correlation_df.columns = ['feature', 'correlation']
    
    fig = px.bar(
        correlation_df,
        x='correlation',
        y='feature',
        orientation='h',
        title='Feature Correlation with Price',
        labels={'x': 'Correlation Coefficient'},
        color='correlation',
        color_continuous_scale='RdBu'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("📍 Price vs Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='sqft_living',
            y='price',
            title='Price vs Living Area',
            trendline='ols',
            labels={'sqft_living': 'Sq Ft Living', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='grade',
            y='price',
            title='Price vs Grade',
            trendline='ols',
            labels={'grade': 'Grade', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FEATURE EXPLORATION ====================
elif page == "Feature Exploration":
    st.title("🔬 Feature Exploration")
    
    st.markdown("---")
    
    # Feature selector
    st.subheader("Select a Feature to Explore")
    
    features = [col for col in df.columns if col not in ['id', 'date', 'lat', 'long']]
    selected_feature = st.selectbox("Choose a feature:", features)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        fig = px.histogram(
            df,
            x=selected_feature,
            title=f'Distribution of {selected_feature}',
            nbins=30,
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # vs Price
        if selected_feature != 'price':
            fig = px.scatter(
                df,
                x=selected_feature,
                y='price',
                title=f'{selected_feature} vs Price',
                trendline='ols',
                color='price',
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.box(
                df,
                y='price',
                title=f'{selected_feature} Distribution',
                color_discrete_sequence=['#636EFA']
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.subheader(f"📊 Statistics for {selected_feature}")
    
    stats = {
        'Count': df[selected_feature].count(),
        'Mean': df[selected_feature].mean(),
        'Median': df[selected_feature].median(),
        'Std Dev': df[selected_feature].std(),
        'Min': df[selected_feature].min(),
        'Max': df[selected_feature].max(),
        'Unique Values': df[selected_feature].nunique()
    }
    
    stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
    st.dataframe(stats_df, use_container_width=True)

# ==================== PAGE: STATISTICS ====================
elif page == "Statistics":
    st.title("📊 Detailed Statistics")
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    summary_stats = df.describe().T
    st.dataframe(summary_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(width=900, height=900)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data info
    st.subheader("Data Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
    
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    st.markdown("---")
    
    # Column details
    st.subheader("Column Details")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Missing': df[col].isnull().sum(),
            'Unique': df[col].nunique()
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)

# ==================== PAGE: ADDRESS LOOKUP ====================
elif page == "Address Lookup":
    st.title("🔍 Address Lookup & Price Comparison")
    st.markdown(
        "Search for multiple property addresses side by side, compare their normalized "
        "results, and run instant price predictions."
    )
    st.caption("Each panel keeps its own search result and prediction so you can compare addresses side by side.")
    st.markdown("---")

    lookup_cols = st.columns(2)
    with lookup_cols[0]:
        render_lookup_slot(0, api_base_url)
    with lookup_cols[1]:
        render_lookup_slot(1, api_base_url)


# ==================== PAGE: LIVE API TESTER ====================
elif page == "Live API Tester":
    st.title("🧪 Live API Tester")
    st.markdown(
        "Automated scenario pipeline runner — tests the full contract chain "
        "(UI → FastAPI → Orchestration → DB) without manual address entry."
    )
    st.caption(f"API target: **{api_base_url}**")

    if "last_prediction_id" not in st.session_state:
        st.session_state["last_prediction_id"] = ""

    def render_response(title: str, status_code, body: dict, target_url: str) -> None:
        st.subheader(title)
        st.write(f"URL: `{target_url}`")
        if status_code is None:
            st.error("Request failed before receiving a response.")
        elif 200 <= status_code < 300:
            st.success(f"Status: {status_code}")
        else:
            st.warning(f"Status: {status_code}")
        st.json(body)

    # ------------------------------------------------------------------ #
    # SECTION 1 — Quick Actions                                           #
    # ------------------------------------------------------------------ #
    st.subheader("Quick Actions")
    qa_col1, qa_col2 = st.columns(2)
    with qa_col1:
        if st.button("Check Health", use_container_width=True):
            sc, body, url = call_api("GET", api_base_url, "/v1/health")
            render_response("Health Response", sc, body, url)
    with qa_col2:
        if st.button("List Recent Predictions", use_container_width=True):
            sc, body, url = call_api("GET", api_base_url, "/v1/predictions", params={"limit": 10})
            render_response("Predictions List Response", sc, body, url)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 2 — Live Validation Runner                                #
    # ------------------------------------------------------------------ #
    st.subheader("Live Validation Runner")
    st.caption(
        "One-click real-address flow: health → normalize → baseline → prediction → detail. "
        "This path is gated by API live readiness by default."
    )

    with st.form("live_validation_form"):
        lv_col1, lv_col2 = st.columns(2)
        with lv_col1:
            lv_line1 = st.text_input("Address Line 1", value="1600 Pennsylvania Ave NW", key="lv_line1")
            lv_city = st.text_input("City", value="Washington", key="lv_city")
            lv_state = st.text_input("State", value="DC", key="lv_state")
        with lv_col2:
            lv_line2 = st.text_input("Address Line 2 (optional)", value="", key="lv_line2")
            lv_postal = st.text_input("Postal Code", value="20500", key="lv_postal")
            lv_country = st.text_input("Country", value="US", key="lv_country")

        lv_requested_by = st.text_input("Requested By", value="live-ui@local", key="lv_requested_by")
        allow_not_ready = st.checkbox(
            "Allow run even when API is not live-ready",
            value=False,
            key="lv_allow_not_ready",
            help="Keep off for strict live-mode validation.",
        )
        lv_submit = st.form_submit_button("Run Live Validation", use_container_width=True)

    if lv_submit:
        live_payload: dict = {
            "address_line_1": lv_line1,
            "city": lv_city,
            "state": lv_state,
            "postal_code": lv_postal,
            "country": lv_country,
        }
        if lv_line2.strip():
            live_payload["address_line_2"] = lv_line2.strip()

        with st.spinner("Running live validation workflow…"):
            health_sc, health_body, health_url = call_api("GET", api_base_url, "/v1/health")

            if health_sc != 200 or not isinstance(health_body, dict):
                render_response("Live Validation • Health", health_sc, health_body, health_url)
            else:
                render_response("Live Validation • Health", health_sc, health_body, health_url)
                live_mode_ready = bool(health_body.get("live_mode_ready", False))
                live_mode_issues = health_body.get("live_mode_issues", [])
                if (not live_mode_ready) and (not allow_not_ready):
                    st.error(
                        "API is not live-ready. Resolve runtime issues first or enable override to continue."
                    )
                    if isinstance(live_mode_issues, list) and live_mode_issues:
                        st.warning("\n".join(f"- {issue}" for issue in live_mode_issues))
                else:
                    norm_sc, norm_body, norm_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/properties/normalize",
                        payload=live_payload,
                    )
                    render_response("Live Validation • Normalize", norm_sc, norm_body, norm_url)

                    baseline_sc, baseline_body, baseline_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/validation/address-baseline",
                        payload=live_payload,
                    )
                    render_response("Live Validation • Baseline", baseline_sc, baseline_body, baseline_url)

                    prediction_payload = dict(live_payload)
                    if lv_requested_by.strip():
                        prediction_payload["requested_by"] = lv_requested_by.strip()

                    pred_sc, pred_body, pred_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/predictions",
                        payload=prediction_payload,
                    )
                    render_response("Live Validation • Create Prediction", pred_sc, pred_body, pred_url)

                    prediction_id = pred_body.get("prediction_id") if isinstance(pred_body, dict) else None
                    if prediction_id:
                        st.session_state["last_prediction_id"] = prediction_id
                        detail_sc, detail_body, detail_url = call_api(
                            "GET",
                            api_base_url,
                            f"/v1/predictions/{prediction_id}",
                        )
                        render_response("Live Validation • Prediction Detail", detail_sc, detail_body, detail_url)

                        if detail_sc == 200 and isinstance(detail_body, dict):
                            provider_responses = detail_body.get("provider_responses", [])
                            if provider_responses:
                                latest_provider = provider_responses[-1]
                                st.info(
                                    "Latest provider response: "
                                    f"{latest_provider.get('provider_name', 'unknown')} "
                                    f"(feature_source={latest_provider.get('feature_source', 'unknown')})"
                                )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 3 — Automated Scenario Pipeline Runner                     #
    # ------------------------------------------------------------------ #
    st.subheader("Automated Scenario Pipeline Runner")
    st.write(
        "Fetch the registered scenario catalog, choose which scenarios to execute, "
        "and run the full audit pipeline for each one in a single request."
    )

    load_col, _spacer = st.columns([1, 3])
    with load_col:
        if st.button("Load Scenarios", use_container_width=True):
            sc, body, _url = call_api("GET", api_base_url, "/v1/validation/scenarios")
            if sc == 200 and isinstance(body, dict):
                st.session_state["scenarios"] = body.get("scenarios", [])
                st.session_state["batch_result"] = None  # clear stale result on reload
                st.success(f"Loaded {len(st.session_state['scenarios'])} scenario(s).")
            else:
                st.error(f"Failed to load scenarios (status {sc}).")
                st.json(body)

    scenarios: list[dict] = st.session_state.get("scenarios", [])

    if not scenarios:
        st.info("Click **Load Scenarios** to fetch the registered catalog from the API.")
    else:
        # Build label → scenario_id map (only registry scenarios; skip live-derived)
        registry_scenarios = [
            s for s in scenarios if not s.get("scenario_id", "").startswith("live-")
        ]
        live_scenarios = [
            s for s in scenarios if s.get("scenario_id", "").startswith("live-")
        ]

        label_to_id: dict[str, str] = {s["label"]: s["scenario_id"] for s in registry_scenarios}

        selected_labels: list[str] = st.multiselect(
            "Select registered scenarios to run",
            options=list(label_to_id.keys()),
            default=list(label_to_id.keys()),
            help="All registered scenarios are selected by default. De-select to skip individual ones.",
        )

        if live_scenarios:
            st.caption(
                f"{len(live_scenarios)} live-derived scenario(s) also available (from recent traffic) "
                "— not shown in multiselect; they run without pre-configured expectations."
            )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            run_selected = st.button(
                "Run Selected Scenarios",
                use_container_width=True,
                disabled=not selected_labels,
            )
        with btn_col2:
            run_all = st.button("Run All", use_container_width=True)

        if run_selected or run_all:
            if run_all:
                batch_payload: dict = {}
            else:
                batch_payload = {"scenario_ids": [label_to_id[l] for l in selected_labels]}

            with st.spinner("Running scenario pipeline batch…"):
                sc, body, _url = call_api(
                    "POST",
                    api_base_url,
                    "/v1/validation/run-scenario-batch",
                    payload=batch_payload,
                )
            st.session_state["batch_result"] = (sc, body)

    # Display batch result (persists across reruns)
    batch_result = st.session_state.get("batch_result")
    if batch_result is not None:
        sc, body = batch_result
        if sc == 200 and isinstance(body, dict):
            total = body.get("total", 0)
            passed = body.get("passed", 0)
            failed = body.get("failed", 0)
            errors = body.get("errors", 0)

            st.markdown("#### Batch Run Summary")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total", total)
            sm2.metric("Passed", passed, delta=None)
            sm3.metric("Failed", failed)
            sm4.metric("Errors", errors)

            results: list[dict] = body.get("results", [])
            if results:
                rows = []
                for r in results:
                    status_icon = {"pass": "✅ PASS", "fail": "❌ FAIL", "error": "⚠️ ERROR"}.get(
                        r.get("pipeline_status", ""), r.get("pipeline_status", "")
                    )
                    rows.append({
                        "Scenario": r.get("label", r.get("scenario_id")),
                        "Category": r.get("category", ""),
                        "Status": status_icon,
                        "Completeness": (
                            f"{r['completeness_score']:.1%}"
                            if r.get("completeness_score") is not None
                            else "—"
                        ),
                        "Price (USD)": (
                            f"${r['predicted_price']:,.0f}"
                            if r.get("predicted_price") is not None
                            else "—"
                        ),
                        "Issues": "; ".join(r.get("issues", [])) or "none",
                        "Error": r.get("error_message") or "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Per-scenario key feature surfacing
                feature_results = [r for r in results if r.get("key_feature_values")]
                if feature_results:
                    with st.expander("Key Feature Values per Scenario", expanded=False):
                        for r in feature_results:
                            kfv: dict = r.get("key_feature_values", {})
                            st.markdown(f"**{r.get('label', r['scenario_id'])}**")
                            kv_cols = st.columns(min(len(kfv), 6) or 1)
                            for idx, (feat, val) in enumerate(kfv.items()):
                                kv_cols[idx % 6].metric(feat, val)

                with st.expander("Raw Batch Response", expanded=False):
                    st.json(body)
        else:
            st.error(f"Batch run failed (status {sc}).")
            st.json(body)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 4 — Ad-Hoc Address Testing                                 #
    # ------------------------------------------------------------------ #
    with st.expander("Ad-Hoc Address & Prediction Testing", expanded=False):
        st.caption(
            "Enter any address to test the pipeline individually. "
            "Use the registered scenarios above for systematic, repeatable testing."
        )

        with st.form("adhoc_address_form"):
            addr_col1, addr_col2 = st.columns(2)
            with addr_col1:
                adhoc_line1 = st.text_input("Address Line 1", value="123 Main St")
                adhoc_city = st.text_input("City", value="Miami")
                adhoc_state = st.text_input("State", value="FL")
            with addr_col2:
                adhoc_line2 = st.text_input("Address Line 2 (optional)", value="")
                adhoc_postal = st.text_input("Postal Code", value="33101")
                adhoc_country = st.text_input("Country", value="US")

            adhoc_requested_by = st.text_input("Requested By (optional)", value="tester@example.com")
            save_btn = st.form_submit_button("Save Payload")

        adhoc_payload: dict = {
            "address_line_1": adhoc_line1,
            "city": adhoc_city,
            "state": adhoc_state,
            "postal_code": adhoc_postal,
            "country": adhoc_country,
        }
        if adhoc_line2.strip():
            adhoc_payload["address_line_2"] = adhoc_line2.strip()
        if save_btn:
            st.info("Ad-hoc payload saved.")
            st.json(adhoc_payload)

        st.markdown("##### Baseline Expectations")
        use_exp = st.checkbox("Apply expectation checks", value=True, key="adhoc_use_exp")
        min_comp = st.slider("Min Completeness", 0.0, 1.0, 0.85, 0.01, disabled=not use_exp)
        req_feats_csv = st.text_input(
            "Required Features (comma-separated)",
            value="BedroomAbvGr,TotRmsAbvGrd,GrLivArea,LotArea",
            disabled=not use_exp,
        )
        use_bounds = st.checkbox("Enforce default feature bounds", value=True, disabled=not use_exp)

        adhoc_expectations = None
        if use_exp:
            req_feats = [f.strip() for f in req_feats_csv.split(",") if f.strip()]
            adhoc_expectations = {
                "min_completeness_score": min_comp,
                "required_features": req_feats,
                "feature_bounds": (
                    {
                        "BedroomAbvGr": {"minimum": 1, "maximum": 8},
                        "TotRmsAbvGrd": {"minimum": 2, "maximum": 14},
                        "GrLivArea": {"minimum": 500, "maximum": 6000},
                        "LotArea": {"minimum": 1000, "maximum": 50000},
                    }
                    if use_bounds
                    else {}
                ),
            }

        adhoc_flow_col1, adhoc_flow_col2 = st.columns(2)
        with adhoc_flow_col1:
            if st.button("Normalize Address", use_container_width=True, key="adhoc_normalize"):
                sc, body, url = call_api("POST", api_base_url, "/v1/properties/normalize", payload=adhoc_payload)
                render_response("Normalize Response", sc, body, url)
        with adhoc_flow_col2:
            if st.button("Create Prediction", use_container_width=True, key="adhoc_predict"):
                pred_payload = dict(adhoc_payload)
                if adhoc_requested_by.strip():
                    pred_payload["requested_by"] = adhoc_requested_by.strip()
                sc, body, url = call_api("POST", api_base_url, "/v1/predictions", payload=pred_payload)
                if isinstance(body, dict) and body.get("prediction_id"):
                    st.session_state["last_prediction_id"] = body["prediction_id"]
                render_response("Create Prediction Response", sc, body, url)

        if st.button("Generate Address Baseline", use_container_width=True, key="adhoc_baseline"):
            bl_payload = dict(adhoc_payload)
            if adhoc_expectations:
                bl_payload["expectations"] = adhoc_expectations
            sc, body, url = call_api("POST", api_base_url, "/v1/validation/address-baseline", payload=bl_payload)
            render_response("Address Baseline Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                kfv = body.get("features", {}).get("key_feature_values", {})
                if isinstance(kfv, dict) and kfv:
                    st.subheader("Key Property Features")
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Bedrooms", kfv.get("BedroomAbvGr", "n/a"))
                    fc2.metric("Total Rooms", kfv.get("TotRmsAbvGrd", "n/a"))
                    fc3.metric("Living SqFt", kfv.get("GrLivArea", "n/a"))
                    fc4, fc5, fc6 = st.columns(3)
                    fc4.metric("Lot Area", kfv.get("LotArea", "n/a"))
                    fc5.metric("Full Baths", kfv.get("FullBath", "n/a"))
                    fc6.metric("Half Baths", kfv.get("HalfBath", "n/a"))
                assessment = body.get("assessment", {})
                checks = assessment.get("checks", []) if isinstance(assessment, dict) else []
                if checks:
                    st.write(f"Overall: **{assessment.get('overall_status', 'unknown')}**")
                    st.dataframe(pd.DataFrame(checks), use_container_width=True)

        if st.button("Run Full Live Audit", use_container_width=True, key="adhoc_full_audit"):
            fa_payload = dict(adhoc_payload)
            if adhoc_requested_by.strip():
                fa_payload["requested_by"] = adhoc_requested_by.strip()
            if adhoc_expectations:
                fa_payload["expectations"] = adhoc_expectations
            sc, body, url = call_api("POST", api_base_url, "/v1/validation/full-audit", payload=fa_payload)
            render_response("Full Live Audit Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                issues = body.get("issues", [])
                if issues:
                    st.warning("\n".join(f"- {i}" for i in issues))
                else:
                    st.success("No audit issues detected.")

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 5 — Policy Simulation                                      #
    # ------------------------------------------------------------------ #
    st.subheader("Policy Simulation")
    st.caption("Training-only utility for policy experiments; not part of live production prediction flow.")

    enable_training_sim = st.checkbox(
        "Enable training simulations",
        value=False,
        help="Keep this off during live-address validation; enable only when exploring policy behavior.",
    )

    if not enable_training_sim:
        st.info(
            "Simulation tools are disabled by default. "
            "Use Normalize/Create Prediction/Full Audit above for live-address validation."
        )
    else:
        with st.form("policy_sim_form"):
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_line1 = st.text_input("Address Line 1", value="413 Duff Ave", key="sim_line1")
                sim_city = st.text_input("City", value="Ames", key="sim_city")
                sim_state = st.text_input("State", value="IA", key="sim_state")
            with sim_col2:
                sim_postal = st.text_input("Postal Code", value="50010", key="sim_postal")
                sim_country = st.text_input("Country", value="US", key="sim_country")
                sim_policies_csv = st.text_input(
                    "Policy Names (comma-separated)",
                    value="balanced-v1,quality-first-v1",
                    key="sim_policies",
                )
            sim_submitted = st.form_submit_button("Simulate Policies")

        if sim_submitted:
            sim_payload: dict = {
                "address_line_1": sim_line1,
                "city": sim_city,
                "state": sim_state,
                "postal_code": sim_postal,
                "country": sim_country,
            }
            policy_names = [p.strip() for p in sim_policies_csv.split(",") if p.strip()]
            if policy_names:
                sim_payload["policy_names"] = policy_names
            sc, body, url = call_api("POST", api_base_url, "/v1/policies/feature/simulate", payload=sim_payload)
            render_response("Policy Simulation Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                sims = body.get("simulations", [])
                if sims:
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Policy": s["policy_name"],
                                "Version": s["policy_version"],
                                "Price ($)": f"${s['predicted_price']:,.0f}",
                                "Completeness": f"{s['completeness_score']:.1%}",
                            }
                            for s in sims
                        ]),
                        use_container_width=True,
                    )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 6 — Prediction Drill-Down                                  #
    # ------------------------------------------------------------------ #
    st.subheader("Prediction Drill-Down")
    prediction_id_input = st.text_input(
        "Prediction ID",
        value=st.session_state.get("last_prediction_id", ""),
        help="Auto-populated after Create Prediction succeeds.",
    )

    drill_col1, drill_col2, drill_col3 = st.columns(3)
    with drill_col1:
        if st.button("Get Detail", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api("GET", api_base_url, f"/v1/predictions/{prediction_id_input.strip()}")
            render_response("Prediction Detail", sc, body, url)
    with drill_col2:
        if st.button("Get Trace", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api("GET", api_base_url, f"/v1/predictions/{prediction_id_input.strip()}/trace")
            render_response("Prediction Trace", sc, body, url)
    with drill_col3:
        if st.button("Get Events", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api(
                "GET",
                api_base_url,
                f"/v1/predictions/{prediction_id_input.strip()}/events",
                params={"limit": 50, "offset": 0, "sort": "desc"},
            )
            render_response("Prediction Events", sc, body, url)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to navigate between different sections of the dashboard.")
