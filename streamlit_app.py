
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# LOAD EXISTING TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "energy_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("⚡ Energy Consumption Predictor")

st.write(
    "Enter the building and energy parameters "
    "to predict energy consumption."
)

st.divider()


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

c1, c2 = st.columns(2)

with c1:

    st.subheader("Building and Energy Parameters")

    temperature = st.number_input(
        "Temperature",
        value=22.0
    )

    humidity = st.number_input(
        "Humidity",
        value=50.0
    )

    occupancy = st.number_input(
        "Occupancy",
        min_value=0,
        value=10,
        step=1
    )

    hvac_usage = st.number_input(
        "HVAC Usage",
        min_value=0.0,
        value=0.0
    )

    sqft = st.number_input(
        "Square Footage",
        min_value=0.0,
        value=1500.0
    )

    lighting_usage = st.number_input(
        "Lighting Usage",
        min_value=0.0,
        value=0.0
    )

    renewable = st.number_input(
        "Renewable Energy",
        min_value=0.0,
        value=0.0
    )

    net_energy = st.number_input(
        "Net Energy",
        value=0.0
    )


with c2:

    st.subheader("Date and Time Parameters")

    date = st.date_input("Date")

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    minute = st.slider(
        "Minute",
        min_value=0,
        max_value=59,
        value=0
    )

    holiday = st.selectbox(
        "Holiday",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    holiday_analysis = st.selectbox(
        "Holiday Analysis",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Energy Consumption", type="primary"):

    try:

        # Construct all 17 expected input features
        row = pd.DataFrame([
            {
                "Hour": hour,
                "Temperature": temperature,
                "Humidity": humidity,
                "Occupancy": occupancy,
                "HVAC usage": hvac_usage,
                "Square Footage": sqft,
                "Lighting Usage": lighting_usage,

                # Provisional weekday representation:
                # verify against the original training data
                "Day of week": date.strftime("%A"),

                "Holiday": holiday,
                "Holiday Analysis": holiday_analysis,
                "Renewable Energy": renewable,
                "Net Energy": net_energy,
                "Temp Occupancy": temperature * occupancy,
                "Year": date.year,
                "Month": date.month,
                "Day": date.day,
                "Minute": minute
            }
        ])

        # --------------------------------------------------
        # FEATURE VALIDATION
        # --------------------------------------------------

        expected = list(model.feature_names_in_)
        received = list(row.columns)

        missing = [
            col for col in expected
            if col not in received
        ]

        extra = [
            col for col in received
            if col not in expected
        ]

        with st.expander("Model Feature Diagnostics"):

            st.write("Expected features:", expected)
            st.write("Received features:", received)
            st.write("Missing features:", missing)
            st.write("Extra features:", extra)

        if missing:

            st.error(
                f"Required model features are missing: {missing}"
            )

            st.stop()

        if extra:

            st.error(
                f"Unexpected input features detected: {extra}"
            )

            st.stop()

        # Match the exact feature order used by the model
        row = row[expected]

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        prediction = model.predict(row)[0]

        st.success(
            f"Predicted Energy Consumption: {prediction:.2f}"
        )

        st.caption(
            "Prediction generated using your existing trained model."
        )

        # Display input summary
        with st.expander("View Prediction Input"):

            st.dataframe(
                row,
                use_container_width=True
            )

    except Exception as e:

        st.error("An error occurred during prediction.")
        st.exception(e)
