# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Streamlit UI
st.title("🌊 AI-Driven Flood Risk Prediction")

rainfall = st.number_input("Enter Rainfall (mm):", min_value=0.0)
river_level = st.number_input("Enter River Water Level (meters):", min_value=0.0)

if st.button("Predict"):
    features = scaler.transform([[rainfall, river_level]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Flood!")
    else:
        st.success("✅ No Flood Risk.")
