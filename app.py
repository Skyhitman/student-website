'''from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_success_model.pkl")  # Load the model

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance_percent = float(request.form["attendance_percent"])
        past_score = float(request.form["past_score"])
        social_media_hours = float(request.form["social_media_hours"])
        
        features = np.array([[study_hours, attendance_percent, past_score, social_media_hours]])
        prediction = model.predict(features)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)'''
import streamlit as st
import pickle
import numpy as np

# Load model
with open("student_success_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Success Predictor")

# Input fields
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0)
attendance = st.number_input("Attendance %", min_value=0, max_value=100, value=80)
past_score = st.number_input("Past Exam Score", min_value=0, max_value=100, value=70)
social_hours = st.number_input("Social Media Hours per Day", min_value=0, max_value=24, value=2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[study_hours, attendance, past_score, social_hours]])
    prediction = model.predict(input_data)[0]
    st.success(f"The student is likely to {'Pass' if prediction==1 else 'Fail'}")
