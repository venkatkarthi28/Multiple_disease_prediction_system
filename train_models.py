# train_models.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

MODEL_DIR = "saved models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Diabetes ---
df_diabetes = pd.read_csv("diabetes.csv")
X = df_diabetes.drop("Outcome", axis=1)
y = df_diabetes["Outcome"]

scaler_d = StandardScaler()
X_scaled = scaler_d.fit_transform(X)

model_d = SVC(kernel='linear', probability=True, class_weight='balanced')
model_d.fit(X_scaled, y)

with open(os.path.join(MODEL_DIR, "diabetes_model.sav"), "wb") as f:
    pickle.dump({"model": model_d, "scaler": scaler_d}, f)

# --- Heart Disease ---
df_heart = pd.read_csv("heart_disease_data.csv")
X = df_heart.drop("target", axis=1)
y = df_heart["target"]

scaler_h = StandardScaler()
X_scaled = scaler_h.fit_transform(X)

model_h = LogisticRegression(max_iter=1000)
model_h.fit(X_scaled, y)

with open(os.path.join(MODEL_DIR, "heart_disease_model.sav"), "wb") as f:
    pickle.dump({"model": model_h, "scaler": scaler_h}, f)

# --- Parkinson's ---
df_park = pd.read_csv("parkinsons.csv")
X = df_park.drop(["name", "status"], axis=1)
y = df_park["status"]

scaler_p = StandardScaler()
X_scaled = scaler_p.fit_transform(X)

model_p = SVC(kernel='rbf', probability=True, class_weight='balanced')
model_p.fit(X_scaled, y)

with open(os.path.join(MODEL_DIR, "parkinson_model.sav"), "wb") as f:
    pickle.dump({"model": model_p, "scaler": scaler_p}, f)

print("All models saved successfully!")