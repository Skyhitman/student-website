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
import pickle
import streamlit as st

# Load model
with open("student_success_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Success Prediction")
# Example input
study_hours = st.number_input("Study Hours", 0, 10)
attendance = st.number_input("Attendance %", 0, 100)
past_score = st.number_input("Past Score", 0, 100)
social_media = st.number_input("Social Media Hours", 0, 10)

if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance, past_score, social_media]])
    st.write("Pass" if prediction[0] == 1 else "Fail")
