from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictor import PortableHousePriceModel


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="KL House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    return PortableHousePriceModel(
        APP_DIR / "xgboost_booster.ubj",
        APP_DIR / "deployment_config.json"
    )


model = load_model()
config = model.config
options = config["categorical_options"]
evaluation = config["evaluation"]

location_options = options["location"]
property_type_options = options["property_type_main"]
furnishing_options = options["furnishing"]

final_test_mape = float(
    evaluation["holdout_mape_percent"]
)


# ============================================================
# EXACT OLD APP VISIBLE STRUCTURE
# Only the model backend and numerical results are updated.
# ============================================================

st.title("KL House Price Predictor")

st.caption(
    "Machine learning-based house price prediction "
    "for Kuala Lumpur residential listings"
)


with st.sidebar:

    st.header("Model Information")

    st.write("Model: XGBoost")

    final_model_summary = pd.DataFrame([{
        "Final Model": "XGBoost",
        "Selection CV RMSE Mean (RM)": 652062.6863,
        "Selection CV MAE Mean (RM)": 279995.6253,
        "Selection CV MAPE Mean (%)": 14.5888,
        "Selection CV R2 Mean": 0.8938,
        "MAE (RM)": 286596.1038,
        "RMSE (RM)": 661168.8913,
        "Test R2": 0.8934,
        "MAPE (%)": 14.6989
    }])

    st.dataframe(
        final_model_summary,
        use_container_width=True
    )


st.subheader("Property Details")

col1, col2, col3 = st.columns(3)


with col1:

    location = st.selectbox(
        "Location",
        location_options
    )

    property_type_main = st.selectbox(
        "Property Type",
        property_type_options
    )

    furnishing = st.selectbox(
        "Furnishing",
        furnishing_options
    )


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


# ============================================================
# FINAL MODEL ADAPTER
#
# The current final model contains additional predictors that
# did not exist in the old UI. To preserve the old interface
# exactly, those additional fields use fixed reference values:
# - size_type = Built-up
# - property_subtype = None
# - additional_rooms = 0
# - is_studio = 0
# ============================================================

property_subtype_options = options["property_subtype"]
property_subtype = (
    "None"
    if "None" in property_subtype_options
    else property_subtype_options[0]
)

input_data = {
    "built_up_sqft": float(size_sqft),
    "land_area_sqft": 0.0,
    "total_rooms": float(rooms_cleaned),
    "additional_rooms": 0.0,
    "is_studio": 0.0,
    "bathrooms": float(bathrooms_cleaned),
    "car_parks": (
        np.nan
        if car_parks_missing == 1
        else float(car_parks_cleaned)
    ),
    "location": location,
    "size_type": "Built-up",
    "property_type_main": property_type_main,
    "property_subtype": property_subtype,
    "furnishing": furnishing
}


if st.button(
    "Predict House Price",
    type="primary"
):

    predicted_price = model.predict(
        input_data
    )

    relative_error = final_test_mape / 100

    lower_bound = (
        predicted_price
        * (1 - relative_error)
    )

    upper_bound = (
        predicted_price
        * (1 + relative_error)
    )

    st.divider()

    st.metric(
        "Predicted House Price",
        f"RM {predicted_price:,.0f}"
    )

    st.write(
        f"Approximate prediction range: "
        f"RM {lower_bound:,.0f} - "
        f"RM {upper_bound:,.0f}"
    )

    st.info(
        f"The prediction range is based on the final model's "
        f"hold-out MAPE of {final_test_mape:.2f}%. "
        "Actual prices may vary due to factors not included in "
        "the dataset, such as renovation quality, floor level, "
        "building age, view, and micro-location."
    )

    with st.expander(
        "Input Summary"
    ):

        st.json({
            "location": location,
            "property_type_main": property_type_main,
            "furnishing": furnishing,
            "size_sqft": size_sqft,
            "rooms_cleaned": rooms_cleaned,
            "bathrooms_cleaned": bathrooms_cleaned,
            "car_parks_cleaned": car_parks_cleaned,
            "car_parks_missing": car_parks_missing
        })
