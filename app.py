from flask import Flask, jsonify, render_template
import numpy as np
import requests
import joblib

app = Flask(__name__)

# -----------------------------
# 🔹 Load AI Model
# -----------------------------
model = joblib.load("landslide_model.pkl")

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
API_KEY = "20590c10e6994758a48505572628"
CITY = "Palakkad"

# -----------------------------
# 🔹 Elevation Function
# -----------------------------
def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url)
        data = response.json()
        return data["results"][0]["elevation"]
    except:
        return 100  # fallback elevation

# -----------------------------
# 🔹 Rainfall Function
# -----------------------------
def get_rainfall():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data.get("rain", {}).get("1h", 0)
    except:
        return 0

# -----------------------------
# 🔹 Micro-Zone Generator
# -----------------------------
def generate_microzones():

    lat_points = np.linspace(LAT_START, LAT_END, ROWS + 1)
    lon_points = np.linspace(LON_START, LON_END, COLS + 1)

    microzones = []

    for i in range(ROWS):
        for j in range(COLS):

            zone = {
                "zone_id": f"Z{i+1}{j+1}",
                "lat1": float(lat_points[i]),
                "lon1": float(lon_points[j]),
                "lat2": float(lat_points[i+1]),
                "lon2": float(lon_points[j+1])
            }

            microzones.append(zone)

    return microzones

# -----------------------------
# 🔹 AI Risk Route
# -----------------------------
@app.route("/grid_risk")
def grid_risk():

    base_grid = generate_microzones()
    rainfall = get_rainfall()

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

        features = [[rain_24h, rain_72h, slope, elevation]]

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
    return "INNOBOT – AI Micro-Zone Landslide System"

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)