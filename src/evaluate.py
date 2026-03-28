

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns


# Load data, model and scaler

DATA_PATH = "Data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)

# Predictions


y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]


# Confusion Matrix

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Classification Report

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# ROC Curve

fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="AUC = %.3f" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
