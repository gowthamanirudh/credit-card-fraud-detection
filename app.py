import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Store test history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit App UI
st.title("Credit Card Fraud Detection")
st.write("Enter values below or simulate a transaction to test fraud detection.")

# Simulated input vectors (30 scaled features: V1 to V28, Time, Amount)
sample_legit = np.random.normal(0, 1, 30)
sample_fraud = np.random.normal(-1, 1, 30)

col1, col2 = st.columns(2)
with col1:
    if st.button("Simulate Legitimate Transaction"):
        input_data = sample_legit
        prediction = model.predict([input_data])[0]
        st.session_state.history.append(prediction)

with col2:
    if st.button("Simulate Fraudulent Transaction"):
        input_data = sample_fraud
        prediction = model.predict([input_data])[0]
        st.session_state.history.append(prediction)

# Custom input form
with st.expander("🔧 Enter Custom Transaction"):
    input_data = []
    for i in range(30):
        val = st.slider(f"Feature {i+1}", -5.0, 5.0, 0.0, step=0.1)
        input_data.append(val)

    if st.button("Predict Custom Input"):
        prediction = model.predict([input_data])[0]
        st.session_state.history.append(prediction)

# Show latest prediction
if "prediction" in locals():
    label = "Fraudulent" if prediction == 1 else "Legitimate"
    st.subheader("🧾 Prediction Result:")
    st.success(f"Transaction is **{label}**")

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie([1], labels=[label], colors=["red" if prediction else "green"], autopct="100%%")
    st.pyplot(fig)

# History line chart
if st.session_state.history:
    st.subheader("📈 Prediction History")
    st.line_chart(st.session_state.history)

# Classification Metrics (static display)
st.subheader("📊 Model Performance")
st.markdown("""
**Legitimate Transactions**
- Precision: 0.9701
- Recall: 0.9898
- F1-Score: 0.9799

**Fraudulent Transactions**
- Precision: 0.9896
- Recall: 0.9695
- F1-Score: 0.9795

**Overall Accuracy**: 97.96%
""")
