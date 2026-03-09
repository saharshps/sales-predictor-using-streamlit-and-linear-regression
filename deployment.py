import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = joblib.load(r"C:\Users\sahar\OneDrive\Desktop\adv_project\multilinear_regression_model")

st.title("Sales Predictor")

tv = st.number_input("TV Advertising", min_value=0)
radio = st.number_input("Radio Advertising", min_value=0)
news = st.number_input("Newspaper Advertising", min_value=0)

if st.button("Predict"):

    input_data = pd.DataFrame([[tv, radio, news]],
                              columns=['TV','Radio','Newspaper'])

    prediction = model.predict(input_data)

    st.success("Prediction Complete")
    st.write("Predicted Sales:", prediction[0])

    
