import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Energy Consumption Predictor")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "energy_model.pkl")

model = load_model()

st.title("⚡ Energy Consumption Predictor")

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
    
    # Inspect the model's expected input features
st.subheader("Model Feature Diagnostics")

expected = list(model.feature_names_in_)
received = list(row.columns)

st.write("Expected model features:", expected)
st.write("Features received:", received)

missing = [col for col in expected if col not in received]
extra = [col for col in received if col not in expected]

st.write("Missing features:", missing)
st.write("Extra features:", extra)

if missing:
    st.error(f"Missing required features: {missing}")
else:
    pred = model.predict(row)[0]
    st.success(f"Predicted energy consumption: {pred:.2f}")
    st.success(f"Predicted energy consumption: {pred:.2f}")
