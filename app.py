from flask import Flask, jsonify, render_template
import numpy as np
import requests
import joblib

app = Flask(__name__)

# -----------------------------
# 🔹 Load AI Model (5 features now)
# -----------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "landslide_model.pkl")

model = joblib.load(model_path)

print("Model expects:", model.n_features_in_)

# -----------------------------
# 🔹 Kavalappara Boundary
# -----------------------------
LAT_START = 10.94
LAT_END   = 10.97
LON_START = 76.23
LON_END   = 76.27

ROWS = 5
COLS = 5

# -----------------------------
# 🔹 Weather API
# -----------------------------
API_KEY = "YOUR_OPENWEATHER_API_KEY"
CITY = "Palakkad"

# -----------------------------
# 🔹 Elevation Cache
# -----------------------------
elevation_cache = {}

def get_elevation(lat, lon):
    key = f"{round(lat,5)}_{round(lon,5)}"

    if key in elevation_cache:
        return elevation_cache[key]

    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=5)
        data = response.json()
        elevation = data["results"][0]["elevation"]
        elevation_cache[key] = elevation
        return elevation
    except:
        return 100  # fallback

# -----------------------------
# 🔹 Rainfall (1-hour)
# -----------------------------
def get_rainfall():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data.get("rain", {}).get("1h", 0)
    except:
        return 0

# -----------------------------
# 🔹 Manual Soil Classification
# -----------------------------
def get_soil_factor(zone):
    """
    Soil factor based on row position.
    Higher value = weaker soil.
    """

    row = zone["row"]

    if row == 1:
        return 0.1   # Rock (stable ridge)
    elif row in [2, 3]:
        return 0.4   # Laterite
    elif row == 4:
        return 0.8   # Clay
    else:
        return 0.9   # Weathered soil (very unstable)

# -----------------------------
# 🔹 Generate Micro-Zones
# -----------------------------
def generate_microzones():

    lat_points = np.linspace(LAT_START, LAT_END, ROWS + 1)
    lon_points = np.linspace(LON_START, LON_END, COLS + 1)

    zones = []

    for i in range(ROWS):
        for j in range(COLS):

            zone = {
                "zone_id": f"Z{i+1}{j+1}",
                "row": i+1,
                "col": j+1,
                "lat1": float(lat_points[i]),
                "lon1": float(lon_points[j]),
                "lat2": float(lat_points[i+1]),
                "lon2": float(lon_points[j+1])
            }

            zones.append(zone)

    return zones

# -----------------------------
# 🔹 AI Grid Risk Route
# -----------------------------
@app.route("/grid_risk")
def grid_risk():

    base_grid = generate_microzones()
    rainfall = get_rainfall()

    # Simple accumulation approximation
    rain_24h = rainfall * 4
    rain_72h = rainfall * 10

    enriched_grid = []
    highest_risk = 0
    most_dangerous = None

    for zone in base_grid:

        center_lat = (zone["lat1"] + zone["lat2"]) / 2
        center_lon = (zone["lon1"] + zone["lon2"]) / 2

        elevation = get_elevation(center_lat, center_lon)

        # Simple slope approximation
        slope = min(abs(elevation - 100) / 300, 1)

        # Soil factor
        soil_factor = get_soil_factor(zone)

        # 🔹 5 Feature Input (VERY IMPORTANT)
        features = [[rain_24h, rain_72h, slope, elevation, soil_factor]]

        probability = model.predict_proba(features)[0][1]
        risk_score = round(float(probability), 2)

        if risk_score < 0.4:
            status = "GREEN"
        elif risk_score < 0.7:
            status = "YELLOW"
        else:
            status = "RED"

        zone["risk"] = risk_score
        zone["status"] = status
        zone["elevation"] = elevation
        zone["slope"] = round(slope, 2)
        zone["soil_factor"] = soil_factor
        zone["rainfall_1h"] = rainfall

        if risk_score > highest_risk:
            highest_risk = risk_score
            most_dangerous = zone["zone_id"]

        enriched_grid.append(zone)

    return jsonify({
        "zones": enriched_grid,
        "most_dangerous": most_dangerous
    })

# -----------------------------
# 🔹 Basic Routes
# -----------------------------
@app.route("/")
def home():
    return "INNOBOT – AI Micro-Zone Landslide System Running"

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)