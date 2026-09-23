import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

# Small demonstration dataset generated from study-related patterns.
# Roll number and class are identification fields only; they are not model inputs.
rng = np.random.default_rng(42)
n = 600

data = pd.DataFrame({
    "study_hours": rng.integers(1, 7, n),
    "attendance": rng.integers(50, 101, n),
    "motivation": rng.integers(1, 6, n),
    "previous_score": rng.integers(35, 96, n),
    "assignments": rng.integers(30, 101, n),
    "study_consistency": rng.integers(1, 6, n),
})

score = (
    data["study_hours"] * 7
    + data["attendance"] * 0.25
    + data["motivation"] * 5
    + data["previous_score"] * 0.45
    + data["assignments"] * 0.15
    + data["study_consistency"] * 4
)

data["performance"] = pd.cut(
    score,
    bins=[-np.inf, 105, 145, np.inf],
    labels=["At Risk", "Medium", "High"]
)

features = [
    "study_hours", "attendance", "motivation",
    "previous_score", "assignments", "study_consistency"
]

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
model.fit(data[features], data["performance"])

st.title("🎓 AI-Based Student Performance Prediction")
st.write("Simple poster-presentation demo using Random Forest.")
st.subheader("Enter Student Information")

col1, col2 = st.columns(2)
with col1:
    roll_no = st.number_input("Roll Number", min_value=1, step=1)
    student_class = st.selectbox("Class", ["FY B.Com IT", "SY B.Com IT", "TY B.Com IT"])
    study_hours = st.slider("Study Hours per Day", 0, 12, 3)
    attendance = st.slider("Attendance (%)", 0, 100, 75)

with col2:
    motivation = st.slider("Motivation Level", 1, 5, 3)
    previous_score = st.slider("Previous Exam Score (%)", 0, 100, 60)
    assignments = st.slider("Assignments Completed (%)", 0, 100, 70)
    study_consistency = st.slider("Study Consistency", 1, 5, 3)

if st.button("Predict Performance"):
    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "motivation": motivation,
        "previous_score": previous_score,
        "assignments": assignments,
        "study_consistency": study_consistency,
    }])

    prediction = model.predict(student)[0]
    confidence = max(model.predict_proba(student)[0]) * 100

    st.subheader("Prediction Result")
    if prediction == "High":
        st.success("🟢 HIGH PERFORMANCE")
    elif prediction == "Medium":
        st.warning("🟡 MEDIUM PERFORMANCE")
    else:
        st.error("🔴 AT RISK")

    st.metric("Prediction Confidence", f"{confidence:.1f}%")
    st.info(
        f"Roll No: {int(roll_no)} | Class: {student_class}\n\n"
        "Roll number and class are for identification only and are not used by the AI model."
    )

st.divider()
st.caption("Educational prototype for poster presentation. Not for official academic decisions.")
