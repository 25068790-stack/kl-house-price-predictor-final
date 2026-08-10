import numpy as np
import streamlit as st
from pathlib import Path

from predictor import PortableHousePriceModel


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="KL House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_bundle():
    model = PortableHousePriceModel(
        APP_DIR / "xgboost_booster.ubj",
        APP_DIR / "deployment_config.json",
    )

    cfg = model.config

    return {
        "model": model,
        "model_name": "XGBoost Tuned",
        "location_options": cfg["categorical_options"]["location"],
        "property_type_options": cfg["categorical_options"]["property_type_main"],
        "furnishing_options": cfg["categorical_options"]["furnishing"],
        "property_subtype_options": cfg["categorical_options"]["property_subtype"],
        "expected_model_features": int(cfg["expected_model_features"]),
    }


bundle = load_bundle()

model = bundle["model"]
model_name = bundle["model_name"]
location_options = bundle["location_options"]
property_type_options = bundle["property_type_options"]
furnishing_options = bundle["furnishing_options"]
property_subtype_options = bundle["property_subtype_options"]

# The visible old page displays this count in Model Summary.
# The deployment refit on all 51,535 observations produces 151
# transformed serving features.
feature_columns = list(
    range(bundle["expected_model_features"])
)

TEST_R2 = 0.893444112290816
TEST_MAPE = 14.698923032337794
MAE_RM = 286596.10380641045
RMSE_RM = 661168.8912831432


def safe_default(options, preferred):
    return preferred if preferred in options else options[0]


def build_input_row(
    location,
    property_type,
    furnishing,
    size_sqft,
    rooms,
    bathrooms,
    car_parks,
    car_parks_missing,
):
    """
    Map the OLD visible interface to the FINAL frozen model.

    The old page does not expose four predictors added to the
    final model, so fixed reference settings are used:
    size_type='Built-up', property_subtype='None',
    additional_rooms=0, is_studio=0.
    """

    property_subtype = (
        "None"
        if "None" in property_subtype_options
        else property_subtype_options[0]
    )

    return {
        "built_up_sqft": float(size_sqft),
        "land_area_sqft": 0.0,
        "total_rooms": float(rooms),
        "additional_rooms": 0.0,
        "is_studio": 0.0,
        "bathrooms": float(bathrooms),
        "car_parks": (
            np.nan
            if car_parks_missing
            else float(car_parks)
        ),
        "location": location,
        "size_type": "Built-up",
        "property_type_main": property_type,
        "property_subtype": property_subtype,
        "furnishing": furnishing,
    }


def predict_price(input_row):
    predicted_price = float(model.predict(input_row))
    mape_ratio = TEST_MAPE / 100
    lower_bound = predicted_price * (1 - mape_ratio)
    upper_bound = predicted_price * (1 + mape_ratio)
    return predicted_price, lower_bound, upper_bound


def format_rm(value):
    return f"RM {value:,.0f}"


st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #111827;
    }

    header, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1260px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f4e79 52%, #0f766e 100%);
        border-radius: 24px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.18);
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: 13px;
        font-weight: 800;
        color: #a7f3d0;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 44px;
        line-height: 1.08;
        font-weight: 900;
        margin-bottom: 14px;
    }

    .hero-copy {
        max-width: 880px;
        color: #dbeafe;
        font-size: 18px;
        line-height: 1.65;
    }

    .section-title {
        font-size: 30px;
        line-height: 1.2;
        font-weight: 900;
        margin-bottom: 8px;
        color: #111827;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 22px;
    }

    .result-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 22px;
        padding: 28px 30px;
        margin-top: 4px;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
    }

    .result-label {
        color: #1d4ed8;
        font-size: 13px;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .result-value {
        color: #0f172a;
        font-size: 48px;
        line-height: 1.05;
        font-weight: 950;
        margin-bottom: 14px;
    }

    .result-range {
        color: #334155;
        font-size: 20px;
        font-weight: 750;
    }

    .empty-card {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 22px;
        padding: 30px;
        margin-top: 4px;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .empty-title {
        color: #111827;
        font-size: 28px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .empty-copy {
        color: #64748b;
        font-size: 17px;
        line-height: 1.6;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 16px;
        margin-bottom: 22px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #111827;
        font-size: 25px;
        font-weight: 900;
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px 22px;
        color: #475569;
        font-size: 15px;
        line-height: 1.65;
        margin-top: 18px;
    }

    .summary-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        margin-top: 8px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        color: #334155;
        line-height: 1.65;
    }

    .stButton button {
        border-radius: 14px;
        height: 3.2rem;
        font-size: 17px;
        font-weight: 800;
        background: #0f766e;
        border-color: #0f766e;
    }

    .stButton button:hover {
        background: #115e59;
        border-color: #115e59;
    }

    .stSelectbox label, .stNumberInput label {
        font-weight: 750;
        color: #374151;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 14px;
        background-color: #ffffff;
        border-color: #dbe3ef;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 32px;
        }

        .result-value {
            font-size: 34px;
        }

        .metric-grid {
            grid-template-columns: 1fr 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Kuala Lumpur Residential Market</div>
        <div class="hero-title">Kuala Lumpur House Price Estimator</div>
        <div class="hero-copy">
            Estimate Kuala Lumpur residential listing prices using a trained machine learning model.
            The output includes a predicted price and an indicative range based on model test error.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


presets = {
    "KLCC serviced residence": {
        "location": "KLCC, Kuala Lumpur",
        "property_type": "Serviced Residence",
        "furnishing": "Fully Furnished",
        "size_sqft": 1200,
        "rooms": 3,
        "bathrooms": 2,
        "car_parks": 1,
        "car_parks_missing": "No",
    },
    "Mont Kiara condominium": {
        "location": "Mont Kiara, Kuala Lumpur",
        "property_type": "Condominium",
        "furnishing": "Partly Furnished",
        "size_sqft": 1400,
        "rooms": 3,
        "bathrooms": 2,
        "car_parks": 2,
        "car_parks_missing": "No",
    },
    "Mid-range apartment": {
        "location": "Setapak, Kuala Lumpur",
        "property_type": "Apartment",
        "furnishing": "Partly Furnished",
        "size_sqft": 900,
        "rooms": 3,
        "bathrooms": 2,
        "car_parks": 1,
        "car_parks_missing": "No",
    },
    "Luxury bungalow": {
        "location": "Damansara Heights, Kuala Lumpur",
        "property_type": "Bungalow",
        "furnishing": "Partly Furnished",
        "size_sqft": 5000,
        "rooms": 5,
        "bathrooms": 5,
        "car_parks": 3,
        "car_parks_missing": "No",
    },
}

if "has_prediction" not in st.session_state:
    st.session_state.has_prediction = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


input_col, result_col = st.columns([1.05, 0.95], gap="large")

with input_col:
    st.markdown('<div class="section-title">Property Inputs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose a preset or enter custom property details.</div>',
        unsafe_allow_html=True,
    )

    preset_name = st.selectbox("Quick scenario", list(presets.keys()))
    preset = presets[preset_name]

    form_col_1, form_col_2 = st.columns(2)

    with form_col_1:
        location = st.selectbox(
            "Location",
            location_options,
            index=location_options.index(safe_default(location_options, preset["location"])),
        )
        property_type = st.selectbox(
            "Property type",
            property_type_options,
            index=property_type_options.index(safe_default(property_type_options, preset["property_type"])),
        )
        furnishing = st.selectbox(
            "Furnishing",
            furnishing_options,
            index=furnishing_options.index(safe_default(furnishing_options, preset["furnishing"])),
        )

    with form_col_2:
        size_sqft = st.number_input(
            "Size (sqft)",
            min_value=300,
            max_value=20000,
            value=preset["size_sqft"],
            step=50,
        )
        rooms = st.number_input(
            "Rooms",
            min_value=1,
            max_value=20,
            value=preset["rooms"],
            step=1,
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=preset["bathrooms"],
            step=1,
        )

    lower_form_col_1, lower_form_col_2 = st.columns(2)

    with lower_form_col_1:
        car_parks = st.number_input(
            "Car parks",
            min_value=0,
            max_value=10,
            value=preset["car_parks"],
            step=1,
        )

    with lower_form_col_2:
        car_parks_missing_choice = st.selectbox(
            "Car park information missing",
            ["No", "Yes"],
            index=0 if preset["car_parks_missing"] == "No" else 1,
        )

    predict_clicked = st.button("Predict House Price", type="primary", use_container_width=True)


if predict_clicked:
    car_parks_missing = car_parks_missing_choice == "Yes"

    input_row = build_input_row(
        location=location,
        property_type=property_type,
        furnishing=furnishing,
        size_sqft=size_sqft,
        rooms=rooms,
        bathrooms=bathrooms,
        car_parks=car_parks,
        car_parks_missing=car_parks_missing,
    )

    predicted_price, lower_bound, upper_bound = predict_price(input_row)

    st.session_state.has_prediction = True
    st.session_state.prediction_result = {
        "predicted_price": predicted_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "location": location,
        "property_type": property_type,
        "furnishing": furnishing,
        "size_sqft": size_sqft,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "car_parks": car_parks,
        "car_parks_missing_choice": car_parks_missing_choice,
    }


with result_col:
    st.markdown('<div class="section-title">Estimated Value</div>', unsafe_allow_html=True)

    if st.session_state.has_prediction and st.session_state.prediction_result is not None:
        st.markdown(
            '<div class="section-subtitle">Estimated price range based on the submitted property details.</div>',
            unsafe_allow_html=True,
        )

        result = st.session_state.prediction_result

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted House Price</div>
                <div class="result-value">{format_rm(result["predicted_price"])}</div>
                <div class="result-range">{format_rm(result["lower_bound"])} - {format_rm(result["upper_bound"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>Input summary:</b><br>
                {result["location"]}<br>
                {result["property_type"]}, {result["furnishing"]}<br>
                {result["size_sqft"]:,.0f} sqft, {result["rooms"]} rooms, {result["bathrooms"]} bathrooms, {result["car_parks"]} car parks
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-subtitle">Run the estimator to generate a price prediction.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="empty-card">
                <div class="empty-title">Ready to estimate</div>
                <div class="empty-copy">
                    Enter property details on the left and click <b>Predict House Price</b>.
                    The model will return a predicted listing price and an indicative price range.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")

st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Core deployment information for the trained model.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Final model</div>
            <div class="metric-value">{model_name}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test R2</div>
            <div class="metric-value">{TEST_R2:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test MAPE</div>
            <div class="metric-value">{TEST_MAPE:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Deployment features</div>
            <div class="metric-value">{len(feature_columns)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="summary-box">
        <b>Model scope:</b> This estimator is based on Kuala Lumpur residential listing data.
        It uses {len(location_options)} location options, {len(property_type_options)} property type options,
        and {len(furnishing_options)} furnishing categories.
        <br><br>
        <b>Interpretation note:</b> The prediction range is derived from the final model's test MAPE.
        Actual transaction prices may vary due to renovation quality, floor level, building age, view,
        tenure, and other micro-location characteristics that are not included in the training data.
    </div>
    """,
    unsafe_allow_html=True,
)
