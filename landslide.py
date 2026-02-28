import requests
import math

# ─── API Keys ─────────────────────────────────────────────────────────────────
WEATHER_API_KEY = '20590c10e6994758a4850557262802'

# ─── Known coordinates for Kerala locations ───────────────────────────────────
LOCATION_COORDS = {
    "Munnar":   (10.0889, 77.0595),
    "Wayanad":  (11.6854, 76.1320),
    "Idukki":   (9.9189,  77.1025),
    "Kochi":    (9.9312,  76.2673),
    "Kottayam": (9.5916,  76.5222),
}

# Soil type & forest density are still static (no free global API exists for these)
# These are research-based values for Kerala districts
STATIC_TERRAIN_DATA = {
    "Munnar":   {"forest_density": 65, "soil_type": "clay"},
    "Wayanad":  {"forest_density": 72, "soil_type": "clay"},
    "Idukki":   {"forest_density": 60, "soil_type": "clay"},
    "Kochi":    {"forest_density": 20, "soil_type": "sandy"},
    "Kottayam": {"forest_density": 45, "soil_type": "loam"},
}

# ─── Open Elevation API ───────────────────────────────────────────────────────
def get_elevation(lat, lon):
    """Fetch elevation in meters for a single coordinate."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    payload = {"locations": [{"latitude": lat, "longitude": lon}]}
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["results"][0]["elevation"]
    except Exception as e:
        print(f"  ⚠ Elevation API error: {e}")
        return None

def calculate_slope(lat, lon):
    """
    Estimate slope (in degrees) by sampling 4 points ~500m N/S/E/W
    of the target and computing the max elevation gradient.
    """
    # ~0.0045 degrees ≈ 500 meters
    offset = 0.0045

    # Sample points: center, north, south, east, west
    points = [
        {"latitude": lat,          "longitude": lon},           # center
        {"latitude": lat + offset, "longitude": lon},           # north
        {"latitude": lat - offset, "longitude": lon},           # south
        {"latitude": lat,          "longitude": lon + offset},  # east
        {"latitude": lat,          "longitude": lon - offset},  # west
    ]

    url = "https://api.open-elevation.com/api/v1/lookup"
    try:
        response = requests.post(url, json={"locations": points}, timeout=15)
        response.raise_for_status()
        elevations = [r["elevation"] for r in response.json()["results"]]

        center, north, south, east, west = elevations
        distance_m = 500  # meters

        # Rise/run for N-S and E-W gradients
        dz_ns = abs(north - south) / (2 * distance_m)
        dz_ew = abs(east - west)  / (2 * distance_m)

        # Resultant gradient magnitude → convert to degrees
        gradient = math.sqrt(dz_ns**2 + dz_ew**2)
        slope_degrees = math.degrees(math.atan(gradient))
        return round(slope_degrees, 2)

    except Exception as e:
        print(f"  ⚠ Slope calculation error: {e}")
        return None

# ─── Weather API ──────────────────────────────────────────────────────────────
def get_weather_data(location):
    url = f'https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Weather API error for {location}: {e}")
        return None

# ─── Map Data (elevation + slope from API, rest from static) ──────────────────
def get_map_data(location):
    coords = LOCATION_COORDS.get(location)
    if not coords:
        print(f"  ⚠ No coordinates found for {location}, using fallback data.")
        return {"slope": 10, "forest_density": 50, "soil_type": "loam", "elevation_m": None}

    lat, lon = coords
    print(f"  → Fetching elevation & slope for {location} ({lat}, {lon})...")

    elevation = get_elevation(lat, lon)
    slope     = calculate_slope(lat, lon)

    # Merge with static soil/forest data
    static = STATIC_TERRAIN_DATA.get(location, {"forest_density": 50, "soil_type": "loam"})

    return {
        "elevation_m":    elevation if elevation is not None else "N/A",
        "slope":          slope     if slope     is not None else 10,   # fallback to 10°
        "forest_density": static["forest_density"],
        "soil_type":      static["soil_type"],
    }

# ─── Risk Calculation ─────────────────────────────────────────────────────────
def calculate_landslide_risk(weather_data, map_data):
    rainfall      = weather_data['current']['precip_mm']
    forest_density = map_data['forest_density']
    soil_type     = map_data['soil_type']
    slope         = map_data['slope']

    risk_score = 0

    # Rainfall
    if rainfall > 50:
        risk_score += 3
    elif rainfall > 20:
        risk_score += 2
    elif rainfall > 0:
        risk_score += 1

    # Forest density
    if forest_density < 30:
        risk_score += 2
    elif forest_density < 60:
        risk_score += 1

    # Soil type
    if soil_type == 'clay':
        risk_score += 3
    elif soil_type == 'sandy':
        risk_score += 1

    # Slope
    if slope > 30:
        risk_score += 3
    elif slope > 15:
        risk_score += 2
    elif slope > 5:
        risk_score += 1

    return risk_score

def risk_label(score):
    if score >= 8:
        return "🔴 HIGH RISK"
    elif score >= 5:
        return "🟠 MODERATE RISK"
    else:
        return "🟢 LOW RISK"

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    locations = ["Munnar", "Wayanad", "Idukki", "Kochi", "Kottayam"]

    print("=" * 55)
    print("   Kerala Landslide Risk Assessment System")
    print("=" * 55)

    for location in locations:
        print(f"\n📍 Processing: {location}")

        weather_data = get_weather_data(location)
        if not weather_data:
            print(f"  ✗ Skipping {location} — weather data unavailable.")
            continue

        map_data = get_map_data(location)
        score    = calculate_landslide_risk(weather_data, map_data)

        print(f"\n  ┌─ Results for {location} " + "─" * (30 - len(location)))
        print(f"  │  Rainfall     : {weather_data['current']['precip_mm']} mm")
        print(f"  │  Elevation    : {map_data['elevation_m']} m")
        print(f"  │  Slope        : {map_data['slope']}°")
        print(f"  │  Soil Type    : {map_data['soil_type']}")
        print(f"  │  Forest Cover : {map_data['forest_density']}%")
        print(f"  │  Risk Score   : {score}/11")
        print(f"  └─ Assessment  : {risk_label(score)}")

    print("\n" + "=" * 55)
    print("  Analysis complete.")
    print("=" * 55)

if __name__ == "__main__":
    main()