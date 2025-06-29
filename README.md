💳 Credit Card Fraud Detection - End-to-End ML Project

This project demonstrates a complete machine learning pipeline to detect fraudulent credit card transactions. It includes data preprocessing, model training, performance evaluation, and an interactive web app built using Streamlit.

🚀 Features

Trained a Random Forest Classifier on the popular credit card fraud detection dataset

Handled class imbalance using undersampling

Feature scaled the Time and Amount columns

Built a Streamlit app for real-time fraud prediction

Included custom input form, real-time prediction pie chart, classification report, confusion matrix, and prediction history line chart

📁 Project Structure

credit_card_fraud_app/
│
├── app.py                   # Streamlit UI application
├── train_model.py          # Script to preprocess data and train the model
├── data/
│   └── creditcard.csv      # Dataset
├── model/
│   ├── fraud_model.pkl     # Trained Random Forest model
│   └── scaler.pkl          # Scaler object (for Time and Amount)
└── README.md               # Project documentation (this file)

📦 Requirements

Install required libraries using pip:

pip install streamlit scikit-learn pandas seaborn matplotlib joblib

⚙️ Model Training (train_model.py)

Loads creditcard.csv

Balances the dataset (492 frauds + 492 legit samples)

Scales Time and Amount using StandardScaler

Trains RandomForestClassifier(class_weight='balanced')

Saves both the model and the scaler using joblib

🖥️ Streamlit App (app.py)

Run the app using:

streamlit run app.py

🔍 Features in the App

Simulate Legit / Fraud Transaction: Uses real samples from dataset to test model

Custom Input Form: Sliders for 28 PCA features (V1 to V28), and inputs for Time and Amount

Real-Time Prediction: After submission, shows prediction (legit/fraud)

Pie Chart: Visual output of predicted label

Line Chart: Tracks your test history

Dashboard: Shows confusion matrix and classification report from test set

📊 Model Performance (on balanced test set)

Legit:
  Precision: 0.9701
  Recall:    0.9898
  F1-score:  0.9799

Fraud:
  Precision: 0.9896
  Recall:    0.9695
  F1-score:  0.9795

Overall Accuracy: 97.96%

✅ The model is highly effective in detecting fraud with very low false positives/negatives.

✅ Highlights

📉 Solves imbalanced classification

💡 Interactive UI for live testing

📊 Visual metrics: confusion matrix, pie chart, line graph

🧠 Ideal for showcasing ML skills on resume

✨ Future Improvements

Add file upload to batch-predict CSVs

Deploy the app using Streamlit Cloud or HuggingFace Spaces

Include prediction probabilities and thresholds

👨‍💻 Author

Made by [Your Name] — feel free to use, fork, and improve!

📎 References

Kaggle Credit Card Fraud Detection Dataset

