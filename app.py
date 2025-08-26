import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# ----------------------------
# Load the trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = load("student_success_model.joblib")
    return model

model = load_model()

# ----------------------------
# App UI
# ----------------------------
st.title("Student Success Predictor 🎓")
st.write("Predict whether a student is likely to pass or fail based on study habits and past performance.")

# Input fields
study_hours = st.number_input("Study hours per day:", min_value=0.0, max_value=24.0, value=3.0)
attendance_percent = st.number_input("Attendance percentage:", min_value=0.0, max_value=100.0, value=75.0)
past_score = st.number_input("Past exam score (out of 100):", min_value=0.0, max_value=100.0, value=60.0)
social_media_hours = st.number_input("Social media hours per day:", min_value=0.0, max_value=24.0, value=2.0)

# Predict button
if st.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        'study_hours': study_hours,
        'attendance_percent': attendance_percent,
        'past_score': past_score,
        'social_media_hours': social_media_hours
    }])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    # Show result
    if prediction == 1:
        st.success(f"The student is likely to PASS ✅ (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f"The student is likely to FAIL ❌ (Confidence: {probability*100:.2f}%)")

# ----------------------------
# Optional: Show example data
# ----------------------------
if st.checkbox("Show example student data"):
    example_data = pd.DataFrame(np.random.rand(5,4)*[6,60,100,6], columns=['study_hours','attendance_percent','past_score','social_media_hours'])
    st.dataframe(example_data)
