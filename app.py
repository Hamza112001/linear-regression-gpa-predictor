import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="GPA Prediction",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎓 GPA Prediction System")

st.write(
    "Enter the student's information below to predict their GPA."
)


# -----------------------------
# Input fields
# -----------------------------
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

assignment_completion = st.number_input(
    "Assignment Completion (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

backlogs = st.number_input(
    "Backlogs",
    min_value=0,
    max_value=20,
    value=0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

motivation = st.number_input(
    "Motivation",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

concentration = st.number_input(
    "Concentration",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

time_management = st.number_input(
    "Time Management",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

self_discipline = st.number_input(
    "Self Discipline",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

procrastination_score = st.number_input(
    "Procrastination Score",
    min_value=0.0,
    max_value=10.0,
    value=3.0
)

previous_gpa = st.number_input(
    "Previous GPA",
    min_value=0.0,
    max_value=4.0,
    value=3.0
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict GPA"):

    # Put inputs in the same order as training data
    input_data = np.array([[
        study_hours,
        attendance,
        assignment_completion,
        backlogs,
        sleep_hours,
        motivation,
        concentration,
        time_management,
        self_discipline,
        procrastination_score,
        previous_gpa
    ]])

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Get predicted GPA
    predicted_gpa = prediction[0]

    st.success(f"Predicted GPA: {predicted_gpa:.2f}")