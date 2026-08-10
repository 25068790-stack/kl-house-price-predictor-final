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
property_subtype_options = options["property_subtype"]
furnishing_options = options["furnishing"]
size_type_options = options["size_type"]

final_test_mape = float(
    evaluation["holdout_mape_percent"]
)


# ============================================================
# Old interface structure retained
# ============================================================

st.title("KL House Price Predictor")

st.caption(
    "Machine learning-based house price prediction "
    "for Kuala Lumpur residential listings"
)


# ------------------------------------------------------------
# Sidebar — same role as the old app
# ------------------------------------------------------------

with st.sidebar:

    st.header("Model Information")

    st.write("Model: XGBoost")

    final_model_summary = pd.DataFrame(
        {
            "Metric": [
                "R²",
                "MAPE (%)",
                "MAE (RM)",
                "RMSE (RM)"
            ],
            "Value": [
                f"{evaluation['holdout_r2']:.4f}",
                f"{evaluation['holdout_mape_percent']:.2f}",
                f"{evaluation['holdout_mae_rm']:,.2f}",
                f"{evaluation['holdout_rmse_rm']:,.2f}"
            ]
        }
    )

    st.dataframe(
        final_model_summary,
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# Property inputs — same plain 3-column layout as the old app
# ------------------------------------------------------------

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

    property_subtype = st.selectbox(
        "Property Subtype",
        property_subtype_options
    )

    furnishing = st.selectbox(
        "Furnishing",
        furnishing_options
    )


with col2:

    size_type = st.selectbox(
        "Size Type",
        size_type_options
    )

    size_sqft = st.number_input(
        "Size (sqft)",
        min_value=300,
        max_value=1000000,
        value=1200,
        step=50
    )

    rooms = st.number_input(
        "Rooms",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

    additional_rooms = st.number_input(
        "Additional Rooms",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    is_studio = st.selectbox(
        "Studio Unit",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col3:

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    bathrooms_missing = st.selectbox(
        "Bathroom Information Missing",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    car_parks = st.number_input(
        "Car Parks",
        min_value=0,
        max_value=15,
        value=1,
        step=1
    )

    car_parks_missing = st.selectbox(
        "Car Park Information Missing",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


# ============================================================
# Build raw input exactly for the final frozen model
# ============================================================

if is_studio == 1:
    rooms = 0
    additional_rooms = 0

total_rooms = rooms

if size_type == "Built-up":
    built_up_sqft = float(size_sqft)
    land_area_sqft = 0.0
else:
    built_up_sqft = 0.0
    land_area_sqft = float(size_sqft)

input_data = {
    "location": location,
    "property_type_main": property_type_main,
    "property_subtype": property_subtype,
    "furnishing": furnishing,
    "size_type": size_type,
    "built_up_sqft": built_up_sqft,
    "land_area_sqft": land_area_sqft,
    "total_rooms": float(total_rooms),
    "additional_rooms": float(additional_rooms),
    "is_studio": float(is_studio),
    "bathrooms": (
        np.nan
        if bathrooms_missing == 1
        else float(bathrooms)
    ),
    "car_parks": (
        np.nan
        if car_parks_missing == 1
        else float(car_parks)
    )
}


# ============================================================
# Prediction — old presentation retained
# ============================================================

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
        "It is shown as an approximate error-context range rather "
        "than a formal statistical prediction interval. "
        "Actual prices may vary due to factors not included in "
        "the dataset, such as renovation quality, floor level, "
        "building age, view, tenure, and micro-location."
    )

    with st.expander(
        "Input Summary"
    ):

        display_input = {
            "Location": location,
            "Property Type": property_type_main,
            "Property Subtype": property_subtype,
            "Furnishing": furnishing,
            "Size Type": size_type,
            "Size (sqft)": size_sqft,
            "Rooms": total_rooms,
            "Additional Rooms": additional_rooms,
            "Studio Unit": (
                "Yes"
                if is_studio == 1
                else "No"
            ),
            "Bathrooms": (
                "Missing"
                if bathrooms_missing == 1
                else bathrooms
            ),
            "Car Parks": (
                "Missing"
                if car_parks_missing == 1
                else car_parks
            )
        }

        st.json(
            display_input
        )
