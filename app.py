```python
import streamlit as st
import joblib
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GPA Prediction System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Section headers */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Prediction box */
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-top: 25px;
    }

    .prediction-label {
        font-size: 18px;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 48px;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        opacity: 0.7;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 GPA Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based GPA prediction using Linear Regression'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📚 Academic Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    assignment_completion = st.number_input(
        "Assignment Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )


with col2:

    backlogs = st.number_input(
        "Number of Backlogs",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    previous_gpa = st.number_input(
        "Previous GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    sleep_hours = st.number_input(
        "Sleep Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )


# ============================================================
# STUDENT BEHAVIOR
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Student Behavior & Skills</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    motivation = st.slider(
        "Motivation",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    concentration = st.slider(
        "Concentration",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


with col2:

    time_management = st.slider(
        "Time Management",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    self_discipline = st.slider(
        "Self Discipline",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


with col3:

    procrastination_score = st.slider(
        "Procrastination",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_button = st.button(
        "🔮 Predict GPA",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # IMPORTANT:
    # The order must be exactly the same as during model training.

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

    # Scale the input using the SAME scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Extract predicted GPA
    predicted_gpa = prediction[0]

    # Keep GPA inside the normal 0-4 range
    predicted_gpa = max(0.0, min(4.0, predicted_gpa))

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-label">Predicted GPA</div>
            <div class="prediction-value">{predicted_gpa:.2f} / 4.00</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "Prediction generated successfully using the trained "
        "Linear Regression model."
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    if predicted_gpa >= 3.5:
        st.info("The predicted GPA is in the higher GPA range.")

    elif predicted_gpa >= 3.0:
        st.info("The predicted GPA is in the good GPA range.")

    elif predicted_gpa >= 2.0:
        st.info("The predicted GPA is in the moderate GPA range.")

    else:
        st.info("The predicted GPA is in the lower GPA range.")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 About the Model</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "Linear Regression"
    )

with col2:
    st.metric(
        "Input Features",
        "11"
    )

with col3:
    st.metric(
        "R² Score",
        "87.49%"
    )


st.markdown(
    """
    This application uses a Linear Regression machine learning model
    trained on student academic and behavioral features to predict GPA.

    Before prediction, the input values are transformed using the same
    StandardScaler that was used during model training.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎓 GPA Prediction System &nbsp;|&nbsp;
        Machine Learning Project &nbsp;|&nbsp;
        Linear Regression
    </div>
    """,
    unsafe_allow_html=True
)
```
