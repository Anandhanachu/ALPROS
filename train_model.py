import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# -----------------------------
# 1️⃣ Generate Synthetic but Realistic Training Data
# -----------------------------

np.random.seed(42)

samples = 2000

rain_24h = np.random.uniform(0, 300, samples)      # mm
rain_72h = np.random.uniform(0, 600, samples)
slope = np.random.uniform(0, 1, samples)
elevation = np.random.uniform(50, 500, samples)

# Logical landslide condition
landslide = (
    (rain_72h > 350) &
    (slope > 0.5)
).astype(int)

data = pd.DataFrame({
    "rain_24h": rain_24h,
    "rain_72h": rain_72h,
    "slope": slope,
    "elevation": elevation,
    "landslide": landslide
})

# -----------------------------
# 2️⃣ Train Model
# -----------------------------

X = data[["rain_24h", "rain_72h", "slope", "elevation"]]
y = data["landslide"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# -----------------------------
# 3️⃣ Save Model
# -----------------------------

joblib.dump(model, "landslide_model.pkl")
print("Model saved successfully.")