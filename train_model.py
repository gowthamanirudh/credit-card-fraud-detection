import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

print("🔹 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# 🔧 Create a perfectly balanced dataset (equal fraud and legit)
print("🔹 Creating balanced dataset (equal legit and fraud)...")
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
print("✅ Balanced dataset shape:", df.shape)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale only 'Time' and 'Amount'
print("🔹 Scaling features...")
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Train-test split
print("🔹 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Train model with class balancing
print("🔹 Training Random Forest with class weights...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model trained.")

# Evaluate
y_pred = model.predict(X_test)
print("\n🔍 Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# Save model and scaler
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and scaler saved to /model")
