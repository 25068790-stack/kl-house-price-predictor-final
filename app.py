from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature function referenced by the exported model pipeline."""
    frame = frame.copy()
    frame["log_built_up_sqft"] = np.log1p(frame["built_up_sqft"])
    frame["log_land_area_sqft"] = np.log1p(frame["land_area_sqft"])
    return frame.drop(columns=["built_up_sqft", "land_area_sqft"])


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(ROOT / "final_xgboost_deployment_pipeline.joblib"),
        joblib.load(ROOT / "deployment_metadata.joblib"),
    )


def selected_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def estimate(pipeline, details: dict[str, object]) -> float:
    record = pd.DataFrame([{
        "built_up_sqft": details["size_sqft"], "land_area_sqft": 0.0,
        "total_rooms": details["rooms"], "additional_rooms": 0.0,
        "is_studio": 0, "bathrooms": details["bathrooms"],
        "car_parks": details["car_parks"], "location": details["location"],
        "size_type": "Built-up", "property_type_main": details["property_type"],
        "property_subtype": "None", "furnishing": details["furnishing"],
    }])
    return float(pipeline.predict(record)[0])


st.set_page_config(page_title="Kuala Lumpur House Price Estimator", page_icon="house", layout="wide")
pipeline, metadata = load_artifacts()
options = metadata["categorical_options"]

st.markdown("""
<style>
.stApp { background: #f5f7fb; color: #111b31; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1840px; padding: 1.25rem 3.65rem 3rem; }
.hero { min-height: 366px; box-sizing: border-box; padding: 61px 64px; border-radius: 36px; color: white; background: linear-gradient(112deg, #101b31 0%, #1a4e76 52%, #18807c 100%); box-shadow: 0 18px 36px rgba(25,50,83,.20); }
.eyebrow { color: #91efcf; font-size: 18px; font-weight: 800; letter-spacing: 3px; margin-bottom: 22px; }
.hero h1 { color: white; font-size: 64px; line-height: 1.08; letter-spacing: 0; margin: 0 0 27px; font-weight: 800; }
.hero p { max-width: 1460px; color: #d5e4f4; font-size: 27px; line-height: 1.68; margin: 0; font-weight: 500; }
.section-title { color: #111b31; font-size: 47px; line-height: 1.15; margin: 52px 0 19px; font-weight: 800; }
.section-copy { color: #6f7b8e; font-size: 23px; margin: 0 0 43px; }
.result-card { min-height: 310px; box-sizing: border-box; border: 2px solid #bfd8ff; border-radius: 38px; padding: 54px 50px; background: #eef6ff; box-shadow: 0 12px 28px rgba(46,93,158,.10); }
.result-label { color: #2757cc; font-size: 19px; font-weight: 800; letter-spacing: 3px; margin-bottom: 22px; }
.result-price { color: #101a34; font-size: 75px; line-height: 1.05; font-weight: 800; margin: 0 0 31px; }
.result-range { color: #34445f; font-size: 30px; line-height: 1.25; font-weight: 750; }
div[data-testid="stWidgetLabel"] p { color: #334158; font-size: 19px; font-weight: 650; }
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { background: #eef1f6; border-radius: 14px; border: 0; color: #30394a; font-size: 18px; }
div[data-testid="stNumberInput"] > div { border-radius: 14px; background: #eef1f6; border: 0; }
.stButton > button { background: #2559cc; color: white; border: 0; border-radius: 13px; font-size: 18px; font-weight: 700; padding: .65rem 1.5rem; }
.stButton > button:hover { background: #1846ad; color: white; }
@media (max-width: 900px) { .block-container { padding: 1rem 1.2rem 2rem; } .hero { min-height: auto; padding: 38px 30px; border-radius: 25px; } .hero h1 { font-size: 42px; } .hero p { font-size: 19px; } .section-title { font-size: 35px; } .result-price { font-size: 48px; } }
</style>
<section class="hero"><div class="eyebrow">KUALA LUMPUR RESIDENTIAL MARKET</div><h1>Kuala Lumpur House Price Estimator</h1><p>Estimate Kuala Lumpur residential listing prices using a trained machine learning model. The output includes a predicted price and an indicative range based on model test error.</p></section>
""", unsafe_allow_html=True)

left, right = st.columns([1.08, 1], gap="large")
with left:
    st.markdown('<div class="section-title">Property Inputs</div><p class="section-copy">Choose a preset or enter custom property details.</p>', unsafe_allow_html=True)
    scenario = st.selectbox("Quick scenario", ["KLCC serviced residence", "Custom property"])
    klcc = scenario == "KLCC serviced residence"
    first, second = st.columns(2, gap="medium")
    with first:
        location = st.selectbox("Location", options["location"], index=selected_index(options["location"], "KLCC, Kuala Lumpur") if klcc else 0)
        property_type = st.selectbox("Property type", options["property_type_main"], index=selected_index(options["property_type_main"], "Serviced Residence") if klcc else 0)
        furnishing = st.selectbox("Furnishing", options["furnishing"], index=selected_index(options["furnishing"], "Fully Furnished") if klcc else 0)
    with second:
        size_sqft = st.number_input("Size (sqft)", min_value=300, max_value=100_000, value=1150 if klcc else 1200, step=50)
        rooms = st.number_input("Rooms", min_value=1, max_value=20, value=1 if klcc else 3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=1 if klcc else 2, step=1)
    car_parks = st.number_input("Car parks", min_value=0, max_value=30, value=1, step=1)
    st.button("Update estimate", type="primary")

with right:
    st.markdown('<div class="section-title">Estimated Value</div><p class="section-copy">Estimated price range based on the submitted property details.</p>', unsafe_allow_html=True)
    prediction = estimate(pipeline, {"location": location, "property_type": property_type, "furnishing": furnishing, "size_sqft": size_sqft, "rooms": rooms, "bathrooms": bathrooms, "car_parks": car_parks})
    error_rate = metadata["evaluation_holdout_mape"] / 100
    st.markdown(f'<section class="result-card"><div class="result-label">PREDICTED HOUSE PRICE</div><div class="result-price">RM {prediction:,.0f}</div><div class="result-range">RM {prediction*(1-error_rate):,.0f} - RM {prediction*(1+error_rate):,.0f}</div></section>', unsafe_allow_html=True)
    st.caption(f"Final XGBoost model. Independent grouped hold-out: RMSE RM {metadata['evaluation_holdout_rmse']:,.0f}; MAPE {metadata['evaluation_holdout_mape']:.2f}%; R2 {metadata['evaluation_holdout_r2']:.3f}.")
