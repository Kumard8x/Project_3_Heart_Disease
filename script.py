import streamlit as st
import joblib
import numpy as np


# Load the trained model
model = joblib.load("rfc_model.pkl")

st.title("Heart Disease Prediction App")
st.image('heart.jpg', width=1000)#set the image.
st.write("Enter patient details to predict heart disease risk.")



col1, col2 = st.columns(2)
# User input fields
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    rest_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
with col2:
    rest_ecg = st.selectbox("Resting ECG Results", ["Normal", "Abnormality", "Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    

# Preprocess user input
input_data = np.array([
    age,
    1 if sex == "Male" else 0,
    ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp),
    rest_bp, cholesterol,
    1 if fbs == "Yes" else 0,
    ["Normal", "Abnormality", "Hypertrophy"].index(rest_ecg),
    max_hr,
    1 if exercise_angina == "Yes" else 0,
    oldpeak,
    ["Upsloping", "Flat", "Downsloping"].index(st_slope)
]).reshape(1, -1)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Heart Disease Detected"
    if prediction == 1:
        st.error(f"No Heart Disease Detected")
    else:
        st.success(f"Prediction : {result}")
    
        

st.markdown(""" ---
    ⚙️ Build By **Deepak Kumar** \n
    📩 **Contact Me:** 
    🔗 [LinkedIn](https://www.linkedin.com/in/deepak-kumar8/) 
    🔗 [GitHub](https://github.com/Kumard8x)
    📧 Email: deepak.kumar030151@gmail.com  
    """)

