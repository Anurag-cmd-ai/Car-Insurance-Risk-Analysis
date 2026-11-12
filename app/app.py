"""Single-screen Streamlit app for car insurance claim risk prediction."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.interpret import feature_importance_df
from src.predict import VALID_CATEGORIES, get_pipeline, predict_single
from src.risk import build_verdict, get_threshold_config


st.set_page_config(
    page_title="Car Insurance Claim Risk Check",
    page_icon="🚗",
    layout="centered",
)

st.title("Car Insurance Claim Risk Predictor")
st.caption(
    "An underwriting-focused estimate of claim likelihood, "
    "based on driver history, vehicle details, and customer profile."
)

age_labels = {0: "16-25", 1: "26-39", 2: "40-64", 3: "65+"}

with st.form("claim_risk_profile"):
    st.subheader("Customer profile")
    demographics, vehicle = st.columns(2)

    with demographics:
        age_display = st.selectbox("Age group", list(age_labels.values()), index=1)
        gender_display = st.selectbox("Gender", ["Female", "Male"])
        education = st.selectbox("Education", VALID_CATEGORIES["education"])
        income = st.selectbox("Income class", VALID_CATEGORIES["income"])
        married_display = st.selectbox("Married", ["No", "Yes"])
        children_display = st.selectbox("Has children", ["No", "Yes"])

    with vehicle:
        driving_experience = st.selectbox(
            "Driving experience", VALID_CATEGORIES["driving_experience"]
        )
        ownership_display = st.selectbox(
            "Vehicle ownership", ["Does not own vehicle", "Owns vehicle"], index=1
        )
        vehicle_year = st.selectbox(
            "Vehicle year", VALID_CATEGORIES["vehicle_year"], index=1
        )
        vehicle_type = st.selectbox("Vehicle type", VALID_CATEGORIES["vehicle_type"])
        credit_score = st.slider("Credit score (normalized)", 0.0, 1.0, 0.52, 0.01)
        annual_mileage = st.slider("Annual mileage", 2000, 22000, 12000, 500)

    st.markdown("**Driving record**")
    speeding_col, dui_col, accident_col = st.columns(3)
    with speeding_col:
        speeding_violations = st.number_input(
            "Speeding violations", min_value=0, max_value=25, value=0, step=1
        )
    with dui_col:
        duis = st.number_input("DUIs", min_value=0, max_value=10, value=0, step=1)
    with accident_col:
        past_accidents = st.number_input(
            "Past accidents", min_value=0, max_value=20, value=0, step=1
        )

    submitted = st.form_submit_button("Assess claim risk", type="primary")

profile = {
    "age": next(code for code, label in age_labels.items() if label == age_display),
    "gender": 0 if gender_display == "Female" else 1,
    "driving_experience": driving_experience,
    "education": education,
    "income": income,
    "credit_score": credit_score,
    "vehicle_ownership": 0 if ownership_display == "Does not own vehicle" else 1,
    "vehicle_year": vehicle_year,
    "married": 0 if married_display == "No" else 1,
    "children": 0 if children_display == "No" else 1,
    "annual_mileage": annual_mileage,
    "vehicle_type": vehicle_type,
    "speeding_violations": speeding_violations,
    "duis": duis,
    "past_accidents": past_accidents,
}

if submitted:
    try:
        prediction = predict_single(profile)
        profile_summary = {
            "Age group": age_display,
            "Gender": gender_display,
            "Driving experience": driving_experience,
            "Education": education,
            "Income class": income,
            "Married": married_display,
            "Has children": children_display,
            "Vehicle ownership": ownership_display,
            "Vehicle year": vehicle_year,
            "Vehicle type": vehicle_type,
            "Credit score": f"{credit_score:.2f}",
            "Annual mileage": f"{annual_mileage:,}",
            "Speeding violations": int(speeding_violations),
            "DUIs": int(duis),
            "Past accidents": int(past_accidents),
        }
        st.session_state["claim_risk_result"] = {
            "prediction": prediction,
            "profile": profile_summary,
        }
        st.session_state.pop("claim_risk_error", None)
    except (ValueError, FileNotFoundError) as error:
        st.session_state.pop("claim_risk_result", None)
        st.session_state["claim_risk_error"] = str(error)
    except Exception as error:
        st.session_state.pop("claim_risk_result", None)
        st.session_state["claim_risk_error"] = (
            f"The assessment could not be completed: {error}"
        )

if st.session_state.get("claim_risk_error"):
    st.error(st.session_state["claim_risk_error"])

saved_result = st.session_state.get("claim_risk_result")
if saved_result:
    prediction = saved_result["prediction"]
    probability = float(prediction["claim_probability"])
    verdict = build_verdict(probability)
    flagged = bool(prediction["predicted_label"])

    st.divider()
    st.subheader("Assessment result")
    probability_col, result_col, band_col = st.columns(3)
    probability_col.metric("Claim probability", verdict["claim_probability_pct"])
    result_col.metric("Claim risk", "Flagged" if flagged else "Not flagged")
    band_col.metric("Risk band", verdict["risk_level"])
    st.progress(probability)

    if flagged:
        st.warning(
            "The model flags this profile as likely to result in a claim "
            "at its 50% classification cutoff."
        )
    else:
        st.success(
            "The model does not flag this profile as likely to result in a claim "
            "at its 50% classification cutoff."
        )

    with st.expander("Input profile summary"):
        st.table(
            pd.DataFrame(
                saved_result["profile"].items(), columns=["Input", "Value"]
            )
        )

    with st.expander("Key predictive factors"):
        factors = feature_importance_df(get_pipeline()).head(10)
        factor_view = pd.DataFrame(
            {
                "Rank": factors["rank"].astype(int).astype(str),
                "Feature": factors["feature"],
                "Coefficient": factors["coefficient"].map(lambda value: f"{value:+.4f}"),
                "Magnitude": factors["importance"].map(lambda value: f"{value:.4f}"),
                "Direction": factors["coefficient"].map(
                    lambda value: "Higher association" if value > 0 else "Lower association"
                ),
            }
        )
        st.table(factor_view)
        st.caption(
            "These model factors show statistical associations, not causes."
        )

    thresholds = get_threshold_config()
    st.caption(
        "Risk bands: "
        f"LOW below {thresholds['low_threshold']:.0%}; "
        f"MEDIUM from {thresholds['low_threshold']:.0%} to below "
        f"{thresholds['high_threshold']:.0%}; "
        f"HIGH at or above {thresholds['high_threshold']:.0%}."
    )

st.divider()
st.caption(
    "This is a statistical estimate based on historical data, not a guarantee "
    "of future claims. Use it as decision support rather than the sole basis "
    "for underwriting."
)