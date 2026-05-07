import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("fare_model.pkl")

st.title("🚕 Taxi Fare Prediction App")

st.write("Enter trip details:")

# =========================
# INPUTS (DROPDOWNS)
# =========================

# Passenger count
passenger_count = st.selectbox(
    "Passenger Count",
    [1, 2, 3, 4, 5, 6]
)

# Trip distance
trip_distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.1,
    value=1.0
)

# Hour dropdown
hour = st.selectbox(
    "Pickup Hour",
    list(range(0, 24))
)

# Weekend dropdown
weekend_option = st.selectbox(
    "Is it Weekend?",
    ["No", "Yes"]
)

is_weekend = 1 if weekend_option == "Yes" else 0

# Night calculation (auto)
is_night = 1 if hour >= 22 or hour <= 5 else 0

# =========================
# PREDICTION
# =========================

if st.button("Predict Fare"):
    
    # Input array (must match model training features)
    input_data = np.array([
        [passenger_count, trip_distance, hour, is_night, is_weekend]
    ])
    
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Fare: ${prediction[0]:.2f}")