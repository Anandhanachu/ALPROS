import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# -----------------------------
# 1️⃣ Generate Synthetic but Realistic Training Data
# -----------------------------

np.random.seed(42)

samples = 3000

rain_24h = np.random.uniform(0, 300, samples)      # mm
rain_72h = np.random.uniform(0, 600, samples)
slope = np.random.uniform(0, 1, samples)
elevation = np.random.uniform(50, 500, samples)

# Soil factor (manual classification simulation)
# 0.1 = rock, 0.4 = laterite, 0.8 = clay, 0.9 = weathered soil
soil_factor = np.random.choice([0.1, 0.4, 0.8, 0.9], samples)

# -----------------------------
# 2️⃣ Logical Landslide Condition (More Realistic)
# -----------------------------

landslide = (
    (rain_72h > 350) &
    (slope > 0.5) &
    (soil_factor > 0.6)   # Weak soil increases failure
).astype(int)

data = pd.DataFrame({
    "rain_24h": rain_24h,
    "rain_72h": rain_72h,
    "slope": slope,
    "elevation": elevation,
    "soil_factor": soil_factor,
    "landslide": landslide
})

# -----------------------------
# 3️⃣ Train Model
# -----------------------------

X = data[["rain_24h", "rain_72h", "slope", "elevation", "soil_factor"]]
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
# 4️⃣ Feature Importance
# -----------------------------

feature_names = ["rain_24h", "rain_72h", "slope", "elevation", "soil_factor"]
importances = model.feature_importances_

print("\nFeature Importances:")
for name, score in zip(feature_names, importances):
    print(f"{name}: {round(score, 3)}")

# -----------------------------
# 5️⃣ Save Model
# -----------------------------

joblib.dump(model, "landslide_model.pkl")
print("\nModel saved successfully.")