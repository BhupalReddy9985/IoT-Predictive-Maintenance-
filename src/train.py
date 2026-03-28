

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


DATA_PATH = "Data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns)


# 2. Split Features and Target

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

print("\nTarget distribution:")
print(y.value_counts())


# 3. Train-Test Split (Stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 4. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for later use in deployment
joblib.dump(scaler, "models/scaler.pkl")

print("\nFeature scaling completed.")


# 5. Baseline Model – Logistic Regression

print("\nTraining Logistic Regression (Baseline)...")

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# 6. Handle Imbalance with SMOTE

print("\nApplying SMOTE to handle class imbalance...")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_sm).value_counts())


# 7. Random Forest Model

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_sm, y_train_sm)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# 8. XGBoost Model (Main Production Model)

print("\nTraining XGBoost (Final Model)...")

# Calculate scale_pos_weight for imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train_sm, y_train_sm)

y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost Results:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# 9. Save Final Model


joblib.dump(xgb, "models/final_model.pkl")

print("\nFinal XGBoost model saved at: models/final_model.pkl")
print("Week 2 model training completed successfully!")
