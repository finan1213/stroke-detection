import streamlit as st
import numpy as np
import joblib

ada_model=joblib.load("healthcare-dataset-stroke-data.pkl")
encoder1=joblib.load("encoder1.pkl")
encoder2=joblib.load("encoder2.pkl")
encoder3=joblib.load("encoder3.pkl")
encoder4=joblib.load("encoder4.pkl")
encoder5=joblib.load("encoder5.pkl")
scaler=joblib.load("scaler.pkl")

st.title("Stroke prediction system")
st.write("Enter patient details below")

gender=st.selectbox("Enter your sex",['female','male'])
age=st.number_input("Enter your age")
hypertension=st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease=st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married=st.selectbox("Ever Married", ['Yes', 'No'])
work_type=st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job'])
Residence_type=st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level=st.number_input("Average Glucose Level")
bmi=st.number_input("BMI")
smoking_status=st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes'])

gender=encoder1.fit_transform([gender])[0]
ever_married=encoder2.fit_transform([ever_married])[0]
work_type=encoder3.fit_transform([work_type])[0]
Residence_type=encoder4.fit_transform([Residence_type])[0]
smoking_status=encoder5.fit_transform([smoking_status])[0]

if st.button("Predict"):
    input_data = np.array([gender, age, hypertension, heart_disease,
                   ever_married, work_type, residence_type,
                   avg_glucose_level, bmi, smoking_status])
    input_scaled = scaler.transform([input_data])
    result = ada_model.predict(input_scaled)
    if result == 1:
        st.error("Stroke Risk Detected!")
    else:
        st.success(" No Stroke Risk Detected.")









