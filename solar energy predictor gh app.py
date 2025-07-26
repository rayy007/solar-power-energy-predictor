import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder

import base64

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


# Title and description
st.title("Solar Power Generation Prediction")
st.write("This app predicts solar power output using an XGBoost model based on meteorological input features.")

# Load the trained XGBoost model
data=pd.read_csv("solarpowergeneration.csv")
avg_power = data['power-generated'].mean()

data['average-wind-speed-(period)']=data['average-wind-speed-(period)'].fillna(data['average-wind-speed-(period)'].median())

data1=data.drop(columns=['power-generated','visibility','average-pressure-(period)'])
for column in data1.columns:
    box=plt.boxplot(data1[column])
    Extremes=[item.get_ydata()[1] for item in box['whiskers']]
    UE=Extremes[1]
    LE=Extremes[0]
    data1.loc[data1[column]>UE,column]=UE
    data1.loc[data1[column]<LE,column]=LE

data.update(data1)
le = LabelEncoder()
data['sky-cover'] = le.fit_transform(data['sky-cover'])

X=data.drop(columns=['power-generated'])
y=data['power-generated']

model_xgb=XGBRegressor(random_state=42, colsample_bytree= 0.8, learning_rate= 0.1, max_depth= 3, n_estimators= 200, subsample= 0.8)
model=model_xgb.fit(X, y)

# Define input fields
st.sidebar.header("Input Features")

def user_input_features():
    distance_solar_noon = st.sidebar.number_input(" Enter the Distance to solar noon (radians)", min_value=0.0, max_value=1.0, value=0.397, step=0.001)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 80.0, 69.0)
    wind_direction =  st.sidebar.slider(" Wind Directon (in degree)", 0,360,28)
    wind_speed =  st.sidebar.slider(" Wind Speed (in m/s)", 0.0,25.0,7.5)
    sky_cover = int(st.sidebar.selectbox('Sky Cover',('0','1','2','3','4')))
    visibility = st.sidebar.slider("Visibility (km)", 0.0, 10.0, 10.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 70.0)
    avg_wind_speed = st.sidebar.slider("Average Wind Speed (m/s)", 0.0, 30.0, 0.0)
    
    pressure = st.sidebar.slider("Pressure (inHg)", 29.4, 30.6, 29.89, step=0.01)

    data = {
	"distance-to-solar-noon" : distance_solar_noon,
        "temperature": temperature,
 	"wind-direction" : wind_direction,
	"wind-speed" :  wind_speed,
 	"sky-cover" : sky_cover,
	"visibility" : visibility, 
	"humidity": humidity,
        "average-wind-speed-(period)": avg_wind_speed,
	"average-pressure-(period)" : pressure
        
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Show user inputs
st.subheader("User Input Parameters")
st.write(input_df)

# Predict
if st.button("Predict"):
	prediction = model.predict(input_df)

# Display prediction
	st.subheader("Predicted Solar Power Output (joules)")
	st.markdown(f"<h3 style='color: orange; font-weight: bold;'>🔶 Predicted Output: {prediction[0]:.2f} joules</h3>", unsafe_allow_html=True)


	fig, ax = plt.subplots()
	ax.bar(['Predicted'], [prediction[0]], color='orange', label='Predicted')
	ax.bar(['Average'], [avg_power], color='skyblue', label='Average')

	ax.set_ylabel("Power Output (kW)")
	ax.set_title("Predicted vs Average Solar Power Output")
	ax.legend()

	st.pyplot(fig)
