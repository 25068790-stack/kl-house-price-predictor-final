from pathlib import Path

import numpy as np
import streamlit as st

from predictor import PortableHousePriceModel


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Kuala Lumpur House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return PortableHousePriceModel(
        APP_DIR / "xgboost_booster.ubj",
        APP_DIR / "deployment_config.json",
    )


model = load_model()
cfg = model.config
options = cfg["categorical_options"]
evaluation = cfg["evaluation"]


st.markdown(
    """
    <style>
      .block-container {
        max-width: 1120px;
        padding-top: 2.0rem;
        padding-bottom: 3rem;
      }
      .hero {
        padding: 1.45rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 1.2rem;
      }
      .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.05rem;
      }
      .hero p {
        margin: 0;
        opacity: .78;
      }
      .result-card {
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-top: .75rem;
      }
      .small-note {
        font-size: .88rem;
        opacity: .72;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
      <h1>Kuala Lumpur House Price Predictor</h1>
      <p>
        Academic research prototype for estimating residential listing prices
        using the final XGBoost model.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


prediction_tab, performance_tab, about_tab = st.tabs(
    ["Price estimate", "Model performance", "About"]
)


with prediction_tab:
    st.subheader("Property details")
    st.caption(
        "Enter the property characteristics below. The output is an estimated "
        "advertised listing price in Malaysian Ringgit (RM)."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")

        with left:
            location = st.selectbox(
                "Location",
                options["location"],
                index=(
                    options["location"].index("Mont Kiara, Kuala Lumpur")
                    if "Mont Kiara, Kuala Lumpur" in options["location"]
                    else 0
                ),
            )

            property_type = st.selectbox(
                "Main property type",
                options["property_type_main"],
                index=(
                    options["property_type_main"].index("Condominium")
                    if "Condominium" in options["property_type_main"]
                    else 0
                ),
            )

            property_subtype = st.selectbox(
                "Property subtype",
                options["property_subtype"],
                index=(
                    options["property_subtype"].index("None")
                    if "None" in options["property_subtype"]
                    else 0
                ),
            )

            furnishing = st.selectbox(
                "Furnishing",
                options["furnishing"],
                index=(
                    options["furnishing"].index("Partly Furnished")
                    if "Partly Furnished" in options["furnishing"]
                    else 0
                ),
            )

        with right:
            size_type = st.radio(
                "Area representation",
                options["size_type"],
                horizontal=True,
            )

            if size_type == "Built-up":
                built_up_sqft = st.number_input(
                    "Built-up area (sq ft)",
                    min_value=300.0,
                    max_value=20000.0,
                    value=1200.0,
                    step=50.0,
                )
                land_area_sqft = 0.0
            else:
                land_area_sqft = st.number_input(
                    "Land area (sq ft)",
                    min_value=300.0,
                    max_value=1000000.0,
                    value=2000.0,
                    step=50.0,
                )
                built_up_sqft = 0.0

            is_studio = st.checkbox("Studio unit", value=False)

            if is_studio:
                main_rooms = 0
                additional_rooms = 0
                st.caption("Room counts are set to 0 for a studio unit.")
            else:
                room_col, add_col = st.columns(2)
                with room_col:
                    main_rooms = st.number_input(
                        "Main rooms",
                        min_value=0,
                        max_value=15,
                        value=3,
                        step=1,
                    )
                with add_col:
                    additional_rooms = st.number_input(
                        "Additional rooms",
                        min_value=0,
                        max_value=10,
                        value=0,
                        step=1,
                    )

            total_rooms = int(main_rooms + additional_rooms)

            bath_col, park_col = st.columns(2)

            with bath_col:
                bathrooms_unknown = st.checkbox(
                    "Bathrooms unknown",
                    value=False,
                )
                bathrooms = (
                    np.nan
                    if bathrooms_unknown
                    else st.number_input(
                        "Bathrooms",
                        min_value=1,
                        max_value=15,
                        value=2,
                        step=1,
                    )
                )

            with park_col:
                car_parks_unknown = st.checkbox(
                    "Car parks unknown",
                    value=False,
                )
                car_parks = (
                    np.nan
                    if car_parks_unknown
                    else st.number_input(
                        "Car parks",
                        min_value=1,
                        max_value=15,
                        value=1,
                        step=1,
                    )
                )

        submitted = st.form_submit_button(
            "Estimate listing price",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if total_rooms > 15:
            st.error(
                "The combined room count must be 15 or fewer to stay within "
                "the modelling range."
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

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Estimated listing price",
                    f"RM {prediction:,.0f}",
                )
                st.markdown(
                    '<div class="small-note">'
                    "Model estimate only — not a professional property valuation."
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                low, high = cfg["training_target_range_rm"]
                if prediction < low or prediction > high:
                    st.warning(
                        "The estimate falls outside the listing-price range used "
                        "in the analytical dataset (RM100,000–RM15,000,000). "
                        "Interpret it with additional caution."
                    )

                with st.expander("Prediction inputs"):
                    st.json(record)

            except Exception as exc:
                st.error("A prediction could not be generated.")
                st.code(str(exc))


with performance_tab:
    st.subheader("Independent hold-out performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "RMSE",
        f"RM {evaluation['holdout_rmse_rm']:,.0f}",
    )
    c2.metric(
        "MAE",
        f"RM {evaluation['holdout_mae_rm']:,.0f}",
    )
    c3.metric(
        "MAPE",
        f"{evaluation['holdout_mape_percent']:.2f}%",
    )
    c4.metric(
        "R²",
        f"{evaluation['holdout_r2']:.4f}",
    )

    st.caption(
        "These metrics are the frozen results from the 10,243-observation "
        "independent hold-out partition. The deployed model was refitted only "
        "after model selection and hold-out evaluation were completed."
    )

    st.info(
        "Final model: XGBoost. Model selection was based on training-only "
        "five-fold grouped cross-validation RMSE. The deployed pipeline was "
        "then refitted on all 51,535 analytical observations without additional "
        "hyperparameter tuning."
    )


with about_tab:
    st.subheader("About this prototype")

    st.write(
        "This application demonstrates the deployment component of a "
        "Kuala Lumpur residential listing-price prediction research project."
    )

    st.write(
        "The model uses property size, room configuration, bathrooms, car parks, "
        "location, property type, property subtype and furnishing status."
    )

    st.write(
        "Predictions refer to advertised listing prices rather than verified "
        "transaction prices."
    )

    st.warning(
        "This prototype is for academic demonstration and should not be used "
        "as a substitute for professional valuation, financial advice, or a "
        "guaranteed transaction price."
    )
