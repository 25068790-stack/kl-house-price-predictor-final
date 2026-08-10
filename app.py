from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "final_xgboost_deployment_pipeline.joblib"
METADATA_PATH = ROOT / "deployment_metadata.joblib"


# Kept at module scope because the exported pipeline's FunctionTransformer
# references __main__.engineer_features when it is loaded by Streamlit.
def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["log_built_up_sqft"] = np.log1p(frame["built_up_sqft"])
    frame["log_land_area_sqft"] = np.log1p(frame["land_area_sqft"])
    return frame.drop(columns=["built_up_sqft", "land_area_sqft"])


@st.cache_resource
def load_artifacts():
    metadata = joblib.load(METADATA_PATH)
    pipeline = joblib.load(MODEL_PATH)
    return pipeline, metadata


st.set_page_config(page_title="KL Listing Price Predictor", page_icon="house", layout="centered")
pipeline, metadata = load_artifacts()

st.title("Kuala Lumpur Listing Price Predictor")
st.caption("Final grouped-validation model: XGBoost")

options = metadata["categorical_options"]
with st.form("prediction_form"):
    location = st.selectbox("Location", options["location"])
    property_type_main = st.selectbox("Property type", options["property_type_main"])
    property_subtype = st.selectbox("Property subtype", options["property_subtype"])
    furnishing = st.selectbox("Furnishing", options["furnishing"])
    size_type = st.selectbox("Area type", ["Built-up", "Land area"])
    size_sqft = st.number_input("Area (sq ft)", min_value=300.0, max_value=1_000_000.0, value=1_200.0, step=50.0)
    total_rooms = st.number_input("Total rooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
    additional_rooms = st.number_input("Additional rooms", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
    car_parks = st.number_input("Car parks", min_value=0.0, max_value=30.0, value=1.0, step=1.0)
    submitted = st.form_submit_button("Predict listing price", type="primary")

if submitted:
    record = pd.DataFrame([{
        "built_up_sqft": size_sqft if size_type == "Built-up" else 0.0,
        "land_area_sqft": size_sqft if size_type == "Land area" else 0.0,
        "total_rooms": total_rooms,
        "additional_rooms": additional_rooms,
        "is_studio": int(total_rooms == 0),
        "bathrooms": bathrooms,
        "car_parks": car_parks,
        "location": location,
        "size_type": size_type,
        "property_type_main": property_type_main,
        "property_subtype": property_subtype,
        "furnishing": furnishing,
    }])
    prediction = float(pipeline.predict(record)[0])
    st.metric("Estimated listing price", f"RM {prediction:,.0f}")

st.divider()
st.caption(
    "Independent grouped hold-out performance: "
    f"RMSE RM {metadata['evaluation_holdout_rmse']:,.0f}; "
    f"MAPE {metadata['evaluation_holdout_mape']:.2f}%; "
    f"R2 {metadata['evaluation_holdout_r2']:.4f}. "
    "This is a predictive demonstration, not a professional valuation."
)
