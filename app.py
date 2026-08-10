
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="KL House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model_bundle():
    metadata = joblib.load("deployment_metadata.joblib")
    pipeline = joblib.load("final_xgboost_deployment_pipeline.joblib")
    return pipeline, metadata


def engineer_features(frame):
    """Required by the final pipeline's exported FunctionTransformer."""
    frame = frame.copy()
    frame["log_built_up_sqft"] = np.log1p(frame["built_up_sqft"])
    frame["log_land_area_sqft"] = np.log1p(frame["land_area_sqft"])
    return frame.drop(columns=["built_up_sqft", "land_area_sqft"])


def build_prediction_input(input_data, metadata):
    subtype_options = metadata["categorical_options"]["property_subtype"]
    return pd.DataFrame([{
        "built_up_sqft": input_data["size_sqft"],
        "land_area_sqft": 0.0,
        "total_rooms": input_data["rooms_cleaned"],
        "additional_rooms": 0.0,
        "is_studio": 0,
        "bathrooms": input_data["bathrooms_cleaned"],
        "car_parks": input_data["car_parks_cleaned"],
        "location": input_data["location"],
        "size_type": "Built-up",
        "property_type_main": input_data["property_type_main"],
        "property_subtype": "None" if "None" in subtype_options else subtype_options[0],
        "furnishing": input_data["furnishing"],
    }])


model, metadata = load_model_bundle()

location_options = metadata["categorical_options"]["location"]
property_type_options = metadata["categorical_options"]["property_type_main"]
furnishing_options = metadata["categorical_options"]["furnishing"]
final_model_summary = pd.DataFrame([{
    "Model": metadata["final_model"],
    "RMSE (RM)": metadata["evaluation_holdout_rmse"],
    "MAE (RM)": metadata["evaluation_holdout_mae"],
    "MAPE (%)": metadata["evaluation_holdout_mape"],
    "R2": metadata["evaluation_holdout_r2"],
}])

st.title("KL House Price Predictor")
st.caption("Machine learning-based house price prediction for Kuala Lumpur residential listings")

with st.sidebar:
    st.header("Model Information")
    st.write(f"Model: {metadata['final_model']}")

    if final_model_summary is not None:
        st.dataframe(final_model_summary, use_container_width=True)

st.subheader("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox("Location", location_options)
    property_type_main = st.selectbox("Property Type", property_type_options)
    furnishing = st.selectbox("Furnishing", furnishing_options)

with col2:
    size_sqft = st.number_input(
        "Size (sqft)",
        min_value=300,
        max_value=20000,
        value=1200,
        step=50
    )
    rooms_cleaned = st.number_input(
        "Rooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )
    bathrooms_cleaned = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col3:
    car_parks_cleaned = st.number_input(
        "Car Parks",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )
    car_parks_missing = st.selectbox(
        "Car Park Information Missing",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

input_data = {
    "location": location,
    "property_type_main": property_type_main,
    "furnishing": furnishing,
    "size_sqft": size_sqft,
    "rooms_cleaned": rooms_cleaned,
    "bathrooms_cleaned": bathrooms_cleaned,
    "car_parks_cleaned": car_parks_cleaned,
    "car_parks_missing": car_parks_missing
}

if st.button("Predict House Price", type="primary"):
    prediction_input = build_prediction_input(input_data, metadata)
    predicted_price = model.predict(prediction_input)[0]

    final_mape = metadata["evaluation_holdout_mape"] / 100
    lower_bound = predicted_price * (1 - final_mape)
    upper_bound = predicted_price * (1 + final_mape)

    st.divider()
    st.metric("Predicted House Price", f"RM {predicted_price:,.0f}")

    st.write(
        f"Approximate prediction range: "
        f"RM {lower_bound:,.0f} - RM {upper_bound:,.0f}"
    )

    st.info(
        f"The prediction range is based on the final model's test MAPE of {metadata['evaluation_holdout_mape']:.2f}%. "
        "Actual prices may vary due to factors not included in the dataset, such as renovation quality, floor level, building age, view, and micro-location."
    )

    with st.expander("Input Summary"):
        st.json(input_data)
