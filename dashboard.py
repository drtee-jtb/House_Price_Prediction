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
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data/raw/Housing.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()


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

# Sidebar
st.sidebar.title("🏠 Housing Dashboard")
st.sidebar.markdown("---")

api_base_url = st.sidebar.text_input(
    "Live API Base URL",
    value=st.session_state.get("api_base_url", "http://127.0.0.1:8000"),
    help="Use the backend URL you want the UI to test against.",
)
st.session_state["api_base_url"] = api_base_url

# Page selection
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Analysis", "Feature Exploration", "Statistics", "Live API Tester"]
)

# ==================== PAGE: OVERVIEW ====================
if page == "Overview":
    st.title("🏠 Housing Price Prediction Dashboard")
    st.markdown("Explore the housing dataset with interactive visualizations")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df), delta=None)
    
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():,.0f}", delta=None)
    
    with col3:
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}", delta=None)
    
    with col4:
        st.metric("Avg Bedrooms", f"{df['bedrooms'].mean():.1f}", delta=None)
    
    st.markdown("---")
    
    # Price distribution
    st.subheader("📊 Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x='price', 
            nbins=50,
            title='Price Distribution',
            labels={'price': 'Price ($)'},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, 
            y='price',
            title='Price Box Plot',
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

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
    # SECTION 2 — Automated Scenario Pipeline Runner                     #
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
    # SECTION 3 — Ad-Hoc Address Testing                                 #
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
    # SECTION 4 — Policy Simulation                                      #
    # ------------------------------------------------------------------ #
    st.subheader("Policy Simulation")
    st.caption("Simulate multiple feature policies against an address to compare predicted prices.")

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
    # SECTION 5 — Prediction Drill-Down                                  #
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
