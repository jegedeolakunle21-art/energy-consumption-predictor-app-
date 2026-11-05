
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Load existing trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "energy_model.pkl"
    return joblib.load(model_path)

model = load_model()

st.title("⚡ Energy Consumption Predictor")
st.write("Enter the required parameters to predict energy consumption.")

# Input fields
c1, c2 = st.columns(2)

with c1:
    temperature = st.number_input("Temperature", value=22.0)
    humidity = st.number_input("Humidity", value=50.0)
    occupancy = st.number_input("Occupancy", value=10, step=1)
    sqft = st.number_input("Square Footage", value=1500.0)
    holiday = st.selectbox("Holiday", [0, 1])
    renewable = st.number_input("Renewable Energy", value=0.0)
    net_energy = st.number_input("Net Energy", value=0.0)

with c2:
    date = st.date_input("Date")
    hour = st.slider("Hour", 0, 23, 12)
    minute = st.slider("Minute", 0, 59, 0)

# Prediction section
if st.button("Predict"):

    # Create input DataFrame FIRST
    row = pd.DataFrame([{
        "Hour": hour,
        "Temperature": temperature,
        "Humidity": humidity,
        "Occupancy": occupancy,
        "Square Footage": sqft,
        "Holiday Analysis": holiday,
        "Renewable Energy": renewable,
        "Net Energy": net_energy,
        "Temp Occupancy": temperature * occupancy,
        "Year": date.year,
        "Month": date.month,
        "Day": date.day,
        "Minute": minute,
    }])

    # Inspect model features AFTER creating row
    st.subheader("Model Feature Diagnostics")

    expected = list(model.feature_names_in_)
    received = list(row.columns)

    missing = [col for col in expected if col not in received]
    extra = [col for col in received if col not in expected]

    st.write("Expected model features:", expected)
    st.write("Features received:", received)
    st.write("Missing features:", missing)
    st.write("Extra features:", extra)

    # Predict only if all required features are present
    if not missing:
        try:
            row = row[expected]
            prediction = model.predict(row)[0]

            st.success(
                f"Predicted energy consumption: {prediction:.2f}"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error(
            "Some required model features are missing. "
            "Send the diagnostic results for correction."
        )
