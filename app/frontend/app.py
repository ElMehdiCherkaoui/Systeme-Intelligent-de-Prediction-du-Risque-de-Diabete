import streamlit as st
import requests


API_URL = "http://api:8000/predict"


with st.sidebar:
    st.write("### Patient Parameters")

    with st.form("prediction_form"):
        Pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=0
        )

        Glucose = st.number_input(
            "Glucose",
            min_value=0, max_value=200, value=0
        )

        BloodPressure = st.number_input(
            "Blood Pressure",
            min_value=0, max_value=150, value=0
        )

        SkinThickness = st.number_input(
            "Skin Thickness",
            min_value=0, max_value=100, value=0
        )

        Insulin = st.number_input(
            "Insulin",
            min_value=0, max_value=900, value=0
        )

        BMI = st.number_input(
            "BMI",
            min_value=0.0, max_value=50.0, value=0.0
        )

        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=2.5, value=0.0
        )

        Age = st.number_input(
            "Age",
            min_value=0, max_value=120, value=0
        )

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

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()
        prediction = result["prediction"]

        if prediction == 1:
            st.error("Higher-risk category")
            st.info(
                "Follow-up: consider professional medical assessment "
                "and appropriate clinical testing."
            )
        else:
            st.success("Lower-risk category")
            st.info(
                "Follow-up: maintain healthy habits and attend routine health check-ups."
            )

        st.caption(
            "This prediction is for educational purposes and is not a medical diagnosis."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")
