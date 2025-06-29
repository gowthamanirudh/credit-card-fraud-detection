# Credit Card Fraud Detection-Mini ML Model Project

This project presents an end-to-end machine learning solution to detect fraudulent credit card transactions using a Random Forest Classifier. It covers all essential stages of a machine learning pipeline including data preprocessing, class balancing, model training, evaluation, and deployment via a Streamlit web app.

The original dataset is highly imbalanced, containing 492 fraud cases out of 284,807 transactions. To address this, the model was trained on a balanced subset using undersampling. The `Time` and `Amount` features were scaled using `StandardScaler`, and PCA-transformed features (`V1` to `V28`) were used for prediction.

### Model Performance (on balanced test set):

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Legit  | 0.9701    | 0.9898 | 0.9799   | 492     |
| Fraud  | 0.9896    | 0.9695 | 0.9795   | 492     |

**Overall Accuracy:** 97.96%

The model demonstrates high precision and recall for both legitimate and fraudulent transactions, minimizing false positives and false negatives.

The accompanying Streamlit app allows users to:
- Simulate predictions using real or custom transaction data
- View model output instantly
- Analyze results with visual tools such as pie charts, confusion matrix, and prediction history line graph

This project showcases practical experience in handling imbalanced datasets, training classification models, evaluating performance, and building interactive ML applications.

The app url is given here: https://credit-card-fraud-detection-b64bcilfsb4lkgejc6a5cn.streamlit.app/
