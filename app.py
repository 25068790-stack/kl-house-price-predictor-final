import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from predictor import PortableHousePriceModel


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="KL House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model_bundle():
    model = PortableHousePriceModel(
        APP_DIR / "xgboost_booster.ubj",
        APP_DIR / "deployment_config.json"
    )

    return {
        "model": model,
        "model_name": "XGBoost Tuned",
        "feature_columns": None,
        "location_options": ['ADIVA Desa ParkCity, Kuala Lumpur', 'Alam Damai, Kuala Lumpur', 'Ampang Hilir, Kuala Lumpur', 'Ampang, Kuala Lumpur', 'Bandar Damai Perdana, Kuala Lumpur', 'Bandar Menjalara, Kuala Lumpur', 'Bandar Sri Damansara, Kuala Lumpur', 'Bandar Tasik Selatan, Kuala Lumpur', 'Bangsar South, Kuala Lumpur', 'Bangsar, Kuala Lumpur', 'Batu Caves, Kuala Lumpur', 'Brickfields, Kuala Lumpur', 'Bukit  Persekutuan, Kuala Lumpur', 'Bukit Bintang, Kuala Lumpur', 'Bukit Damansara, Kuala Lumpur', 'Bukit Jalil, Kuala Lumpur', 'Bukit Kiara, Kuala Lumpur', 'Bukit Ledang, Kuala Lumpur', 'Bukit Tunku (Kenny Hills), Kuala Lumpur', 'Chan Sow Lin, Kuala Lumpur', 'Cheras, Kuala Lumpur', 'City Centre, Kuala Lumpur', 'Country Heights Damansara, Kuala Lumpur', 'Damansara Heights, Kuala Lumpur', 'Damansara, Kuala Lumpur', 'Desa Pandan, Kuala Lumpur', 'Desa ParkCity, Kuala Lumpur', 'Desa Petaling, Kuala Lumpur', 'Dutamas, Kuala Lumpur', 'Federal Hill, Kuala Lumpur', 'Gombak, Kuala Lumpur', 'Gurney, Kuala Lumpur', 'Jalan Ipoh, Kuala Lumpur', 'Jalan Klang Lama (Old Klang Road), Kuala Lumpur', 'Jalan Kuching, Kuala Lumpur', 'Jalan Sultan Ismail, Kuala Lumpur', 'Jalan U-Thant, Kuala Lumpur', 'Jinjang, Kuala Lumpur', 'KL City, Kuala Lumpur', 'KL Eco City, Kuala Lumpur', 'KL Sentral, Kuala Lumpur', 'KLCC, Kuala Lumpur', 'Kemensah, Kuala Lumpur', 'Kepong, Kuala Lumpur', 'Keramat, Kuala Lumpur', 'Klcc, Kuala Lumpur', 'Kota Damansara, Kuala Lumpur', 'Kuala Lumpur, Kuala Lumpur', 'Kuchai Lama, Kuala Lumpur', 'Landed Sd, Kuala Lumpur', 'Mid Valley City, Kuala Lumpur', 'Mont Kiara, Kuala Lumpur', 'OUG, Kuala Lumpur', 'Off Gasing Indah,, Kuala Lumpur', 'Other, Kuala Lumpur', 'Pandan Indah, Kuala Lumpur', 'Pandan Jaya, Kuala Lumpur', 'Pandan Perdana, Kuala Lumpur', 'Pantai, Kuala Lumpur', 'Petaling Jaya, Kuala Lumpur', 'Puchong, Kuala Lumpur', 'Rawang, Kuala Lumpur', 'SANTUARI PARK PANTAI, Kuala Lumpur', 'SEMARAK, Kuala Lumpur', 'Salak Selatan, Kuala Lumpur', 'Santuari Park Pantai, Kuala Lumpur', 'Segambut, Kuala Lumpur', 'Sentul, Kuala Lumpur', 'Seputeh, Kuala Lumpur', 'Seri Kembangan, Kuala Lumpur', 'Setapak, Kuala Lumpur', 'Setiawangsa, Kuala Lumpur', 'Solaris Dutamas, Kuala Lumpur', 'Sri Damansara, Kuala Lumpur', 'Sri Hartamas, Kuala Lumpur', 'Sri Kembangan, Kuala Lumpur', 'Sri Petaling, Kuala Lumpur', 'Sungai Besi, Kuala Lumpur', 'Sungai Long SL8, Kuala Lumpur', 'Sungai Penchala, Kuala Lumpur', 'Sunway SPK, Kuala Lumpur', 'Taman Desa, Kuala Lumpur', 'Taman Duta, Kuala Lumpur', 'Taman Ibukota, Kuala Lumpur', 'Taman Melawati, Kuala Lumpur', 'Taman Sri Keramat, Kuala Lumpur', 'Taman TAR, Kuala Lumpur', 'Taman Tun Dr Ismail, Kuala Lumpur', 'Taman Wangsa Permai, Kuala Lumpur', 'Taman Yarl OUG, Kuala Lumpur', 'Taman Yarl, Kuala Lumpur', 'Taman Yarl, UOG, Kuala Lumpur', 'Titiwangsa, Kuala Lumpur', 'Ukay Heights, Kuala Lumpur', 'Wangsa Maju, Kuala Lumpur', 'duta Nusantara, Kuala Lumpur', 'kepong, Kuala Lumpur', 'taman cheras perdana, Kuala Lumpur'],
        "property_type_options": ['1-sty Terrace/Link House', '1.5-sty Terrace/Link House', '2-sty Terrace/Link House', '2.5-sty Terrace/Link House', '3-sty Terrace/Link House', '3.5-sty Terrace/Link House', '4-sty Terrace/Link House', '4.5-sty Terrace/Link House', 'Apartment', 'Bungalow', 'Bungalow Land', 'Cluster House', 'Condominium', 'Flat', 'Residential Land', 'Semi-detached House', 'Serviced Residence', 'Townhouse'],
        "furnishing_options": ['Fully Furnished', 'Partly Furnished', 'Unfurnished', 'Unknown'],
        "final_model_summary": pd.DataFrame(
            [{'Final Model': 'XGBoost Tuned', 'MAE (RM)': 286596.1, 'RMSE (RM)': 661168.89, 'Train R2': 0.8938, 'Test R2': 0.8934, 'R2 Gap': 0.0004, 'MAPE (%)': 14.7}]
        ),
    }


def build_prediction_input(input_data, feature_columns):
    # Adapter from the exact OLD visible inputs to the FINAL 12-predictor model.
    # Extra final-model predictors that did not exist in the old UI are fixed
    # to reference values so the visible interface remains unchanged.
    return {
        "built_up_sqft": float(input_data["size_sqft"]),
        "land_area_sqft": 0.0,
        "total_rooms": float(input_data["rooms_cleaned"]),
        "additional_rooms": 0.0,
        "is_studio": 0.0,
        "bathrooms": float(input_data["bathrooms_cleaned"]),
        "car_parks": (
            np.nan
            if input_data["car_parks_missing"] == 1
            else float(input_data["car_parks_cleaned"])
        ),
        "location": input_data["location"],
        "size_type": "Built-up",
        "property_type_main": input_data["property_type_main"],
        "property_subtype": "None",
        "furnishing": input_data["furnishing"],
    }


bundle = load_model_bundle()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
location_options = bundle["location_options"]
property_type_options = bundle["property_type_options"]
furnishing_options = bundle["furnishing_options"]
final_model_summary = bundle["final_model_summary"]

st.title("KL House Price Predictor")
st.caption("Machine learning-based house price prediction for Kuala Lumpur residential listings")

with st.sidebar:
    st.header("Model Information")
    st.write(f"Model: {bundle['model_name']}")

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
    prediction_input = build_prediction_input(input_data, feature_columns)
    predicted_price = model.predict(prediction_input)

    lower_bound = predicted_price * (1 - 0.146989)
    upper_bound = predicted_price * (1 + 0.146989)

    st.divider()
    st.metric("Predicted House Price", f"RM {predicted_price:,.0f}")

    st.write(
        f"Approximate prediction range: "
        f"RM {lower_bound:,.0f} - RM {upper_bound:,.0f}"
    )

    st.info(
        "The prediction range is based on the final model's test MAPE of 14.70%. "
        "Actual prices may vary due to factors not included in the dataset, such as renovation quality, floor level, building age, view, and micro-location."
    )

    with st.expander("Input Summary"):
        st.json(input_data)
