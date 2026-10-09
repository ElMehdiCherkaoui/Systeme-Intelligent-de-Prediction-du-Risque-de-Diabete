import streamlit as st
import pandas as pd
import joblib 

model_path = "notebooks/best_model.joblib"

with st.sidebar:
    st.write("### Patient Parameters")
    
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
    Insulin = st.number_input("Insulin", min_value=0, max_value=200, value=0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
    Age = st.number_input("Age", min_value=0, max_value=120, value=0)
    
    submit = st.form_submit_button("Predict Diabetes")

if submit:
    input_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    input_df = pd.DataFrame([input_data])
    
    try:
        model = joblib.load(model_path)
        predection = model.predict(input_df)[0]
        st.success(f"Model loaded successfully! The connection works. Prediction: {predection}")
    except Exception as e:
        st.error(f"Error loading model: {e}")