import streamlit as st
import pandas as pd
from joblib import load

model = load('model_XGBoost.joblib')
scaler = load('scaler.joblib')

features = ['zn','chas','rm','age','dis','rad','tax','ptratio','b','lstat']

st.title("Boston House Price Predictor")

inputs = []

inputs.append(st.slider('ZN', 0.0, 100.0, 10.0))
inputs.append(st.selectbox('CHAS', [0,1]))
inputs.append(st.slider('RM', 3.0, 9.0, 6.0))
inputs.append(st.slider('AGE', 0.0, 100.0, 65.0))
inputs.append(st.slider('DIS', 1.0, 12.0, 4.0))
inputs.append(st.slider('RAD', 1, 24, 5))
inputs.append(st.slider('TAX', 100, 700, 300))
inputs.append(st.slider('PTRATIO', 10.0, 22.0, 18.0))
inputs.append(st.slider('B', 0.0, 400.0, 350.0))
inputs.append(st.slider('LSTAT', 1.0, 40.0, 12.0))

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=features)

    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]

    st.success(f"Predicted Price: ${pred*1000:,.0f}")