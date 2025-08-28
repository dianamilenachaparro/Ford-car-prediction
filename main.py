import streamlit as st
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

st.title("🚗Ford Car Price Prediction")

st.divider()

st.write("Con esta app, tu puedes estimar el precio de tu carro marca Ford de acuerdo a las diferentes características")

modelo = ['Fiesta','Focus','Puma','Kuga','EcoSport','C-MAX','Mondeo','Ka+',
 'Tourneo Custom','S-MAX','B-MAX','Edge','Tourneo Connect','Grand C-MAX',
 'KA' 'Galaxy','Mustang','Grand Tourneo Connect','Fusion','Ranger',
 'Streetka','Escort','Transit Tourneo']
transmissionn = ['Automatic', 'Manual', 'Semi-Auto']
fuelTypee = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']

model_mapping = {value: i for i, value in enumerate(modelo)}
transmission_mapping = {value: i for i, value in enumerate(transmissionn)}
fuelType_mapping = {value: i for i, value in enumerate(fuelTypee)}


model = st.selectbox("Modelo", ['Fiesta','Focus','Puma','Kuga','EcoSport','C-MAX','Mondeo','Ka+',
 'Tourneo Custom','S-MAX','B-MAX','Edge','Tourneo Connect','Grand C-MAX',
 'KA' 'Galaxy','Mustang','Grand Tourneo Connect','Fusion','Ranger',
 'Streetka','Escort','Transit Tourneo'])
year = st.number_input("Año", value=1996, step = 1, min_value=1996, max_value=2025)
transmission = st.selectbox("Tipo de transmisión", ['Automatic', 'Manual', 'Semi-Auto'])
mileage = st.number_input("Kilometros", value = 10000, step = 1, min_value=0)
fuelType = st.selectbox("Combustible", ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'])
tax = st.number_input("Impuesto en dolares", value = 150, step= 1, min_value=0)
mpg = st.number_input("Consumo kilometros por galón", value = 92, step = 1, min_value=1)
engineSize = st.number_input("Tamaño del motor", value = 2, min_value=0, max_value=5)


#X = [model_mapping[model],year, transmission_mapping[transmission], mileage, fuelType_mapping[fuelType], tax, mpg, engineSize]
X = pd.DataFrame([{
    "model":model,
    "year":year, 
    "transmission":transmission, 
    "mileage":mileage, 
    "fuelType":fuelType, 
    "tax":tax, 
    "mpg":mpg, 
    "engineSize":engineSize}])

model = joblib.load("best_xgb_model.pkl")

st.divider()

price = st.button("Precio")

st.divider()
if price:
    st.balloons()
    #X1 = np.array([X])

    prediction = model.predict(X)[0]
    print(prediction)

    st.write(f"El precio en dólares del vehículo es ${prediction:,.2f}")
