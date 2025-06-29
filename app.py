import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")
df = pd.read_csv("data/creditcard.csv")

# Prepare real samples
a_sample_legit = df[df['Class'] == 0].iloc[0].drop("Class").values
scaled_legit = a_sample_legit.copy()
scaled_legit[0:2] = scaler.transform([a_sample_legit[0:2]])[0]

a_sample_fraud = df[df['Class'] == 1].iloc[0].drop("Class").values
scaled_fraud = a_sample_fraud.copy()
scaled_fraud[0:2] = scaler.transform([a_sample_fraud[0:2]])[0]

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Credit Card Transaction Checker")

# --- Simulate Real Transactions ---
st.subheader("🔍 Simulate Real Transaction")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Simulate Legit Transaction"):
        prediction = model.predict([scaled_legit])[0]
        if prediction == 1:
            st.error("⚠️ Incorrectly flagged as Fraud (False Positive)")
        else:
            st.success("✅ Legitimate Transaction")

with col2:
    if st.button("🚨 Simulate Fraud Transaction"):
        prediction = model.predict([scaled_fraud])[0]
        if prediction == 1:
            st.success("✅ Correctly Detected Fraud")
        else:
            st.error("❌ Missed Fraud (False Negative)")

st.markdown("---")

# --- Custom Test Form ---
st.subheader("🧪 Test Your Own Transaction")
time = st.number_input("⏱️ Time (seconds since first transaction)", value=10000.0)
amount = st.number_input("💰 Amount ($)", value=150.0)

feature_names = [f"V{i}" for i in range(1, 29)]
custom_features = [st.slider(f"{name}", -5.0, 5.0, 0.0) for name in feature_names]

# Prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Predict Custom Transaction"):
    scaled_time, scaled_amount = scaler.transform([[time, amount]])[0]
    input_data = [scaled_time, scaled_amount] + custom_features
    prediction = model.predict([input_data])[0]
    st.session_state.prediction_history.append(prediction)

    if prediction == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    # Pie chart to visualize prediction
    fig, ax = plt.subplots()
    labels = ['Legit', 'Fraud']
    sizes = [1 - prediction, prediction]
    colors = ['green', 'red']
    explode = (0, 0.1) if prediction == 1 else (0.1, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

st.markdown("---")

# --- Model Metrics Dashboard ---
st.subheader("📊 Model Evaluation Dashboard")
# Load balanced data used for training again
df_bal = pd.concat([
    df[df['Class'] == 0].sample(n=492, random_state=42),
    df[df['Class'] == 1]
])
df_bal = df_bal.sample(frac=1, random_state=42)

X_bal = df_bal.drop("Class", axis=1)
y_bal = df_bal["Class"]
X_bal[["Time", "Amount"]] = scaler.transform(X_bal[["Time", "Amount"]])

preds = model.predict(X_bal)
report = classification_report(y_bal, preds, output_dict=True, target_names=["Legit", "Fraud"])

st.write("**Classification Report:**")
st.json(report)

st.write("**Confusion Matrix:**")
cm = confusion_matrix(y_bal, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
st.pyplot(fig)

# --- Prediction History Line Chart ---
st.subheader("📈 Prediction History (Your Inputs)")
if st.session_state.prediction_history:
    pred_df = pd.DataFrame({"Prediction #": list(range(1, len(st.session_state.prediction_history)+1)),
                            "Prediction": st.session_state.prediction_history})
    pred_df["Label"] = pred_df["Prediction"].map({0: "Legit", 1: "Fraud"})
    st.line_chart(pred_df["Prediction"], height=250, use_container_width=True)
else:
    st.info("No predictions made yet.")