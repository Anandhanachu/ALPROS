from flask import Flask, jsonify, render_template, request
import numpy as np
import requests
import joblib
import os

app = Flask(__name__)

# -----------------------------
# 🔹 Load AI Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "landslide_model.pkl")

model = joblib.load(model_path)
print("Model expects:", model.n_features_in_)

# -----------------------------
# 🔹 Region Definitions
# Each region has:
#   lat/lon bounding box, OpenWeather city name, display name
# -----------------------------
REGIONS = {
    "kavalappara": {
        "name":      "Kavalappara",
        "lat_start": 10.94,
        "lat_end":   10.97,
        "lon_start": 76.23,
        "lon_end":   76.27,
        "city":      "Palakkad",   # nearest OWM city
    },
    "munnar": {
        "name":      "Munnar",
        "lat_start": 10.06,
        "lat_end":   10.10,
        "lon_start": 77.04,
        "lon_end":   77.08,
        "city":      "Munnar",
    },
    "wayanad": {
        "name":      "Wayanad",
        "lat_start": 11.60,
        "lat_end":   11.64,
        "lon_start": 76.07,
        "lon_end":   76.11,
        "city":      "Kalpetta",   # district HQ – better OWM coverage
    },
    "malappuram": {
        "name":      "Malappuram",
        "lat_start": 11.08,
        "lat_end":   11.12,
        "lon_start": 76.07,
        "lon_end":   76.11,
        "city":      "Malappuram",
    },
}

ROWS = 5
COLS = 5

# -----------------------------
# 🔹 API Keys & Config
# -----------------------------
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

BHUVAN_API_TOKEN  = "1f104bc7e2d0512b22d1c7dae877f61ec58a6218"
BHUVAN_WMS_URL    = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
BHUVAN_LULC_LAYER = "lulc50k_1516"

# -----------------------------
# 🔹 LULC Class → Soil Risk Factor
# -----------------------------
LULC_RISK_MAP = {
    "Built-up":                        0.65,
    "Agricultural Land":               0.55,
    "Forest":                          0.25,
    "Wasteland":                       0.80,
    "Water Bodies":                    0.30,
    "Grassland / Grazing Land":        0.50,
    "Scrub Land":                      0.75,
    "Snow and Glaciers":               0.20,
    "Barren / Rocky / Stony Waste":   0.15,
    "Plantations":                     0.30,
    "Mining / Industrial":             0.70,
    "DEFAULT":                         0.50,
}

# -----------------------------
# 🔹 Caches (keyed by lat/lon so all regions share one cache)
# -----------------------------
elevation_cache = {}
soil_cache      = {}
rainfall_cache  = {}   # keyed by city name

# -----------------------------
# 🔹 Get Elevation (Open-Elevation)
# -----------------------------
def get_elevation(lat, lon):
    key = f"{round(lat, 5)}_{round(lon, 5)}"
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
# 🔹 Get Rainfall (OpenWeather) — per city
# -----------------------------
def get_rainfall(city):
    if city in rainfall_cache:
        return rainfall_cache[city]
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        rainfall = data.get("rain", {}).get("1h", 0)
        rainfall_cache[city] = rainfall
        return rainfall
    except:
        return 0

# -----------------------------
# 🔹 Get Soil Factor from Bhuvan WMS
# -----------------------------
def get_soil_factor_bhuvan(lat, lon):
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    if key in soil_cache:
        return soil_cache[key]

    try:
        delta = 0.0005
        bbox  = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"

        params = {
            "SERVICE":      "WMS",
            "VERSION":      "1.1.1",
            "REQUEST":      "GetFeatureInfo",
            "LAYERS":       BHUVAN_LULC_LAYER,
            "QUERY_LAYERS": BHUVAN_LULC_LAYER,
            "STYLES":       "",
            "BBOX":         bbox,
            "WIDTH":        "3",
            "HEIGHT":       "3",
            "SRS":          "EPSG:4326",
            "FORMAT":       "image/png",
            "INFO_FORMAT":  "application/json",
            "X":            "1",
            "Y":            "1",
            "token":        BHUVAN_API_TOKEN,
        }

        response = requests.get(BHUVAN_WMS_URL, params=params, timeout=8)

        if response.status_code == 200:
            try:
                data     = response.json()
                features = data.get("features", [])
                if features:
                    props = features[0].get("properties", {})
                    lulc_class = (
                        props.get("class_name")
                        or props.get("Class_Name")
                        or props.get("LULC_CLASS")
                        or props.get("lulc_class")
                        or props.get("category")
                        or ""
                    )
                    factor = LULC_RISK_MAP["DEFAULT"]
                    for class_key, risk in LULC_RISK_MAP.items():
                        if class_key.lower() in lulc_class.lower():
                            factor = risk
                            break
                    soil_cache[key] = factor
                    print(f"Bhuvan LULC at ({lat},{lon}): '{lulc_class}' → soil_factor={factor}")
                    return factor
            except ValueError:
                factor = _parse_lulc_text(response.text.strip())
                soil_cache[key] = factor
                return factor

        print(f"Bhuvan WMS returned status {response.status_code} for ({lat},{lon})")

    except requests.exceptions.Timeout:
        print(f"Bhuvan WMS timeout for ({lat},{lon})")
    except Exception as e:
        print(f"Bhuvan WMS error for ({lat},{lon}): {e}")

    return 0.5


def _parse_lulc_text(text):
    text_lower = text.lower()
    for class_key, risk in LULC_RISK_MAP.items():
        if class_key.lower() in text_lower:
            return risk
    return LULC_RISK_MAP["DEFAULT"]


# -----------------------------
# 🔹 Generate Micro-Zones for a given region
# -----------------------------
def generate_microzones(region_cfg):
    lat_points = np.linspace(region_cfg["lat_start"], region_cfg["lat_end"], ROWS + 1)
    lon_points = np.linspace(region_cfg["lon_start"], region_cfg["lon_end"], COLS + 1)

    zones = []
    for i in range(ROWS):
        for j in range(COLS):
            zones.append({
                "zone_id": f"Z{i+1}{j+1}",
                "row":     i + 1,
                "col":     j + 1,
                "lat1":    float(lat_points[i]),
                "lon1":    float(lon_points[j]),
                "lat2":    float(lat_points[i + 1]),
                "lon2":    float(lon_points[j + 1]),
            })
    return zones


# -----------------------------
# 🔹 AI Grid Risk Route
# Usage: /grid_risk?region=munnar
#        /grid_risk           (defaults to kavalappara)
# -----------------------------
@app.route("/grid_risk")
def grid_risk():
    region_key = request.args.get("region", "kavalappara").lower().strip()

    if region_key not in REGIONS:
        return jsonify({
            "error":            f"Unknown region '{region_key}'.",
            "available_regions": list(REGIONS.keys()),
        }), 400

    region_cfg = REGIONS[region_key]
    base_grid  = generate_microzones(region_cfg)
    rainfall   = get_rainfall(region_cfg["city"])

    rain_24h = rainfall * 4
    rain_72h = rainfall * 10

    enriched_grid  = []
    highest_risk   = 0
    most_dangerous = None

    for zone in base_grid:
        center_lat = (zone["lat1"] + zone["lat2"]) / 2
        center_lon = (zone["lon1"] + zone["lon2"]) / 2

        elevation   = get_elevation(center_lat, center_lon)
        slope       = min(abs(elevation - 100) / 300, 1)
        soil_factor = get_soil_factor_bhuvan(center_lat, center_lon)

        features    = [[rain_24h, rain_72h, slope, elevation, soil_factor]]
        probability = model.predict_proba(features)[0][1]
        risk_score  = round(float(probability), 2)

        if risk_score < 0.4:
            status = "GREEN"
        elif risk_score < 0.7:
            status = "YELLOW"
        else:
            status = "RED"

        zone["risk"]        = risk_score
        zone["status"]      = status
        zone["elevation"]   = elevation
        zone["slope"]       = round(slope, 2)
        zone["soil_factor"] = soil_factor
        zone["rainfall_1h"] = rainfall

        if risk_score > highest_risk:
            highest_risk   = risk_score
            most_dangerous = zone["zone_id"]

        enriched_grid.append(zone)

    return jsonify({
        "region":         region_cfg["name"],
        "region_key":     region_key,
        "zones":          enriched_grid,
        "most_dangerous": most_dangerous,
    })


# -----------------------------
# 🔹 List available regions
# -----------------------------
@app.route("/regions")
def list_regions():
    return jsonify({
        k: {"name": v["name"], "city": v["city"]}
        for k, v in REGIONS.items()
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


if __name__ == "__main__":
    app.run(debug=True)