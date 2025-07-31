import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64

# -------------------------
# Set background image
# -------------------------
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("solarpic2.png")

# -------------------------
# App title and description
# -------------------------
st.title("S☀️lar Power Generation Predictor")
st.write("Enter environmental conditions to predict solar power output.")

# -------------------------
# Load model and data
# -------------------------
model = joblib.load("cb_model.pkl")
data = pd.read_csv("SolarPowerGen.csv")  # This should be your preprocessed CSV
avg_power = data["power-generated"].mean()

# -------------------------
# User input section
# -------------------------
st.sidebar.header("Input Features")

def user_input_features():
    distance_solar_noon = st.sidebar.number_input("Distance to Solar Noon (radians)", 0.0, 1.0, 0.397, 0.001)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 80.0, 69.0)
    wind_direction = st.sidebar.slider("Wind Direction (°)", 0, 360, 28)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 7.5)
    sky_cover = int(st.sidebar.selectbox("Sky Cover", ('0','1','2','3','4')))
    visibility = st.sidebar.slider("Visibility (km)", 0.0, 10.0, 10.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 70.0)
    avg_wind_speed = st.sidebar.slider("Avg Wind Speed (m/s)", 0.0, 30.0, 0.0)
    pressure = st.sidebar.slider("Pressure (inHg)", 29.4, 30.6, 29.89, 0.01)

    return pd.DataFrame([{
        "distance-to-solar-noon": distance_solar_noon,
        "temperature": temperature,
        "wind-direction": wind_direction,
        "wind-speed": wind_speed,
        "sky-cover": sky_cover,
        "visibility": visibility,
        "humidity": humidity,
        "average-wind-speed-(period)": avg_wind_speed,
        "average-pressure-(period)": pressure
    }])

input_df = user_input_features()
st.subheader("User Input Parameters")
st.write(input_df)
# -------------------------
# Predict and display results
# -------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]

    # Styled output
    st.markdown(
        f"<h3 style='color: yellow; font-weight: bold;'>🔶 Predicted Output: {prediction:.2f} joules</h3>",
        unsafe_allow_html=True
    )
