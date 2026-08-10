from pathlib import Path

import numpy as np
import streamlit as st

from predictor import PortableHousePriceModel


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="KL House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    return PortableHousePriceModel(
        APP_DIR / "xgboost_booster.ubj",
        APP_DIR / "deployment_config.json",
    )


def default_index(items, preferred):
    return items.index(preferred) if preferred in items else 0


model = load_model()
cfg = model.config
options = cfg["categorical_options"]
evaluation = cfg["evaluation"]


# ------------------------------------------------------------
# Light-touch layout cleanup.
# The overall structure intentionally follows the earlier app:
# title + caption, model information in the sidebar,
# three-column property input area, then one prediction result.
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 340px;
        }

        h1 {
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 650;
        }

        .prediction-note {
            color: #5f6670;
            font-size: 0.92rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar — follows the earlier Streamlit prototype structure
# ============================================================
with st.sidebar:
    st.header("Model Information")

    st.write("**Final model:** XGBoost")
    st.write("**Deployment sample:** 51,535 observations")
    st.write("**Model selection:** Training-only grouped CV RMSE")

    st.divider()
    st.subheader("Independent Hold-out")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            "RMSE",
            f"RM {evaluation['holdout_rmse_rm']:,.0f}",
        )
        st.metric(
            "MAPE",
            f"{evaluation['holdout_mape_percent']:.2f}%",
        )

    with metric_col2:
        st.metric(
            "MAE",
            f"RM {evaluation['holdout_mae_rm']:,.0f}",
        )
        st.metric(
            "R²",
            f"{evaluation['holdout_r2']:.4f}",
        )

    st.caption(
        "Performance shown above is from the frozen "
        "10,243-observation independent hold-out partition."
    )

    st.divider()
    st.caption(
        "Academic research prototype for Kuala Lumpur residential "
        "listing-price prediction. Predictions are not professional valuations."
    )


# ============================================================
# Main page
# ============================================================
st.title("KL House Price Predictor")
st.caption(
    "Machine learning-based listing price prediction for "
    "Kuala Lumpur residential properties"
)

st.divider()
st.subheader("Property Details")
st.write(
    "Enter the property's characteristics below, then select "
    "**Predict House Price**."
)


with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # --------------------------------------------------------
    # Column 1 — location / property identity
    # --------------------------------------------------------
    with col1:
        st.markdown("#### Property Profile")

        location = st.selectbox(
            "Location",
            options["location"],
            index=default_index(
                options["location"],
                "Mont Kiara, Kuala Lumpur",
            ),
        )

        property_type = st.selectbox(
            "Property Type",
            options["property_type_main"],
            index=default_index(
                options["property_type_main"],
                "Condominium",
            ),
        )

        property_subtype = st.selectbox(
            "Property Subtype",
            options["property_subtype"],
            index=default_index(
                options["property_subtype"],
                "None",
            ),
        )

        furnishing = st.selectbox(
            "Furnishing",
            options["furnishing"],
            index=default_index(
                options["furnishing"],
                "Partly Furnished",
            ),
        )

    # --------------------------------------------------------
    # Column 2 — area / rooms
    # --------------------------------------------------------
    with col2:
        st.markdown("#### Size & Rooms")

        size_type = st.radio(
            "Area Type",
            options["size_type"],
            horizontal=True,
        )

        if size_type == "Built-up":
            built_up_sqft = st.number_input(
                "Built-up Size (sq ft)",
                min_value=300.0,
                max_value=20000.0,
                value=1200.0,
                step=50.0,
            )
            land_area_sqft = 0.0
        else:
            land_area_sqft = st.number_input(
                "Land Area (sq ft)",
                min_value=300.0,
                max_value=1000000.0,
                value=2000.0,
                step=50.0,
            )
            built_up_sqft = 0.0

        is_studio = st.checkbox(
            "Studio unit",
            value=False,
        )

        if is_studio:
            main_rooms = 0
            additional_rooms = 0
            st.caption("Room counts are set to 0 for a studio unit.")
        else:
            main_rooms = st.number_input(
                "Main Rooms",
                min_value=0,
                max_value=15,
                value=3,
                step=1,
            )

            additional_rooms = st.number_input(
                "Additional Rooms",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )

    # --------------------------------------------------------
    # Column 3 — bathrooms / parking
    # --------------------------------------------------------
    with col3:
        st.markdown("#### Facilities")

        bathroom_status = st.selectbox(
            "Bathroom Information",
            ["Available", "Missing"],
            index=0,
        )

        if bathroom_status == "Available":
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=15,
                value=2,
                step=1,
            )
        else:
            bathrooms = np.nan
            st.caption(
                "Missing bathroom information will be handled by "
                "the fitted preprocessing pipeline."
            )

        car_park_status = st.selectbox(
            "Car Park Information",
            ["Available", "Missing"],
            index=0,
        )

        if car_park_status == "Available":
            car_parks = st.number_input(
                "Car Parks",
                min_value=0,
                max_value=15,
                value=1,
                step=1,
            )
        else:
            car_parks = np.nan
            st.caption(
                "Missing car-park information will be handled by "
                "the fitted preprocessing pipeline."
            )

        st.write("")
        st.caption(
            "The application uses the same frozen predictor structure "
            "as the final research model."
        )

    total_rooms = int(main_rooms + additional_rooms)

    st.write("")
    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Prediction result
# ============================================================
if submitted:
    if total_rooms > 15:
        st.error(
            "The combined room count must be 15 or fewer to remain "
            "within the modelling range."
        )
    else:
        record = {
            "built_up_sqft": float(built_up_sqft),
            "land_area_sqft": float(land_area_sqft),
            "total_rooms": float(total_rooms),
            "additional_rooms": float(additional_rooms),
            "is_studio": float(bool(is_studio)),
            "bathrooms": bathrooms,
            "car_parks": car_parks,
            "location": location,
            "size_type": size_type,
            "property_type_main": property_type,
            "property_subtype": property_subtype,
            "furnishing": furnishing,
        }

        try:
            prediction = model.predict(record)

            mape_rate = evaluation["holdout_mape_percent"] / 100.0
            lower_bound = prediction * (1.0 - mape_rate)
            upper_bound = prediction * (1.0 + mape_rate)

            st.divider()
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(
                [1.45, 1],
                gap="large",
            )

            with result_col1:
                st.metric(
                    "Estimated Listing Price",
                    f"RM {prediction:,.0f}",
                )

                st.write(
                    "**Illustrative MAPE-based range:** "
                    f"RM {lower_bound:,.0f} – RM {upper_bound:,.0f}"
                )
                st.caption(
                    "This range uses the model's hold-out MAPE only as "
                    "an error-context illustration; it is not a formal "
                    "statistical prediction interval."
                )

            with result_col2:
                st.metric(
                    "Hold-out MAPE",
                    f"{evaluation['holdout_mape_percent']:.2f}%",
                )
                st.caption(
                    "Final XGBoost hold-out performance: "
                    f"RMSE RM {evaluation['holdout_rmse_rm']:,.0f}, "
                    f"MAE RM {evaluation['holdout_mae_rm']:,.0f}, "
                    f"R² {evaluation['holdout_r2']:.4f}."
                )

            low, high = cfg["training_target_range_rm"]
            if prediction < low or prediction > high:
                st.warning(
                    "This estimate falls outside the listing-price range "
                    "used in the analytical dataset "
                    "(RM100,000–RM15,000,000). Interpret it with extra caution."
                )
            else:
                st.info(
                    "Actual listing prices may vary because some market and "
                    "property-specific factors are not available in the dataset, "
                    "such as building age, floor level, renovation quality, "
                    "view, tenure and detailed micro-location."
                )

            with st.expander("Input Summary"):
                st.json(record)

        except Exception as exc:
            st.error("A prediction could not be generated.")
            st.code(str(exc))


st.divider()
st.caption(
    "Academic demonstration only. The estimated value refers to an "
    "advertised residential listing price and should not be interpreted "
    "as a guaranteed transaction price or professional valuation."
)
