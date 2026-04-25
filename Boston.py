import streamlit as st
import pandas as pd
from joblib import load

# Load model and scaler
model  = load('model_XGBoost.joblib')
scaler = load('scaler.joblib')

st.title("🏠 Boston House Price Predictor")

# Inputs
zn      = st.slider('ZN — % large residential lots',       0.0,  100.0, 10.0)
indus   = st.slider('INDUS — % non-retail business acres', 0.0,  30.0,  10.0)
chas    = st.selectbox('CHAS — Bounds Charles River?',     [0, 1])
rm      = st.slider('RM — Avg rooms per dwelling',         3.0,  9.0,   6.0)
age     = st.slider('AGE — % units built before 1940',     0.0,  100.0, 65.0)
dis     = st.slider('DIS — Distance to employment centres',1.0,  12.0,  4.0)
rad     = st.slider('RAD — Highway accessibility index',   1,    24,    5)
tax     = st.slider('TAX — Property tax rate',             100,  700,   300)
ptratio = st.slider('PTRATIO — Pupil-teacher ratio',       10.0, 22.0,  18.0)
b       = st.slider('B — B indicator',                     0.0,  400.0, 350.0)
lstat   = st.slider('LSTAT — % lower status population',   1.0,  40.0,  12.0)

# Predict
if st.button('Predict Price'):
    input_data = pd.DataFrame(
        [[zn, indus, chas, rm, age, dis, rad, tax, ptratio, b, lstat]],
        columns=['zn', 'indus', 'chas', 'rm', 'age', 'dis', 
                 'rad', 'tax', 'ptratio', 'b', 'lstat']
    )

    scaled_input = scaler.transform(input_data)
    prediction   = model.predict(scaled_input)

    st.success(f"🏠 Predicted House Price: ${prediction[0]*1000:,.0f}")