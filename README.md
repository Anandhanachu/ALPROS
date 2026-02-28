# ALPROS - AI-Powered Landslide Risk Assessment System

ALPROS is a comprehensive landslide risk prediction and monitoring system designed for Kerala's vulnerable regions. It combines classical statistical analysis with state-of-the-art machine learning to provide accurate, real-time landslide hazard assessments across micro-scale geographic zones.

## System Components

ALPROS features three integrated modules:

1. **CLI Risk Analyzer** (`landslide.py`) - Traditional rule-based assessment
2. **ML Risk Predictor** (`train_model_with_soil.py` & `landslide_model.pkl`) - XGBoost-based classification
3. **Web Dashboard** (`app.py`) - Real-time micro-zone grid visualization

## Features

### Real-time Analysis
- **Live Weather Integration**: Fetches current precipitation via OpenWeatherMap API
- **Elevation & Slope Calculation**: Uses Open Elevation API for precise topographic analysis
- **Soil Factor Classification**: Distinguishes between rock, laterite, clay, and weathered soil

### AI-Powered Risk Assessment
- **Machine Learning Model**: XGBoost classifier trained on 3,000+ synthetic realistic scenarios
- **5-Factor Risk Evaluation**:
  - 24-hour rainfall accumulation
  - 72-hour rainfall accumulation
  - Terrain slope angle (0-1 normalized scale)
  - Digital elevation data
  - Soil composition factor (0.1-0.9 scale)
- **Probability-based Scoring**: Returns landslide probability (0.0-1.0)

### Micro-Zone Grid System
- **Divided Coverage**: Geographic area segmented into 5×5 micro-zones
- **Color-coded Risk Levels**:
  - 🟢 **GREEN** - Low risk (probability < 0.4)
  - 🟡 **YELLOW** - Moderate risk (probability 0.4-0.7)
  - 🔴 **RED** - High risk (probability ≥ 0.7)
- **Zone-specific Insights**: Elevation, slope, soil type per zone

## Supported Locations

Currently monitors the following Kerala districts:
- **Munnar** - High forest density, clay soil, mountainous terrain
- **Wayanad** - Dense forests, clay soil, elevated region
- **Idukki** - Moderate forest cover, clay soil, hilly terrain
- **Kochi** - Low forest density, sandy soil, coastal area
- **Kottayam** - Medium forest coverage, loam soil

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for API calls)

### Required Libraries

```bash
pip install requests flask xgboost scikit-learn joblib pandas numpy
```

### API Keys Required

1. **WeatherAPI Key** or **OpenWeatherMap API Key**
   - Get free key at [weatherapi.com](https://www.weatherapi.com/) or [openweathermap.org](https://openweathermap.org/api)
   - Update `WEATHER_API_KEY` in `landslide.py` or `API_KEY` in `app.py`

2. **Open Elevation API**: Free, no authentication required
   - Automatically fetches elevation data from [open-elevation.com](https://open-elevation.com/)

### Installation

Clone the repository:
```bash
git clone <repository-url>
cd ALPROS
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install requests flask xgboost scikit-learn joblib pandas numpy

## Usage

### 1. Train the Machine Learning Model

Before running the web app, train the XGBoost model:

```bash
python train_model_with_soil.py
```

**Output:**
```
Model Accuracy: 0.9467

Feature Importances:
rain_24h: 0.185
rain_72h: 0.312
slope: 0.288
elevation: 0.124
soil_factor: 0.091

Model saved successfully.
```

This creates `landslide_model.pkl` (the trained classifier).

### 2. Run CLI Risk Assessment

For single-location batch analysis:

```bash
python landslide.py
```

### 3. Run Web Dashboard (AI Micro-Zone System)

Start the Flask development server:

```bash
python app.py
```

Access the dashboard:
- **Web UI**: http://localhost:5000/dashboard
- **API Endpoint**: http://localhost:5000/grid_risk (returns JSON)

#### Sample Web API Response

```json
{
  "zones": [
    {
      "zone_id": "Z11",
      "row": 1,
      "col": 1,
      "lat1": 10.94,
      "lon1": 76.23,
      "lat2": 10.946,
      "lon2": 76.243,
      "risk": 0.15,
      "status": "GREEN",
      "elevation": 125,
      "slope": 0.08,
      "soil_factor": 0.1,
      "rainfall_1h": 0.0
    },
    {
      "zone_id": "Z25",
      "row": 2,
      "col": 5,
      "lat1": 10.946,
      "lon1": 76.27,
      "lat2": 10.952,
      "lon2": 76.283,
      "risk": 0.78,
      "status": "RED",
      "elevation": 245,
      "slope": 0.48,
      "soil_factor": 0.8,
      "rainfall_1h": 12.5
    }
  ],
  "most_dangerous": "Z25"
}
```

---

### CLI Output Example

When running `landslide.py`:

```
===========================================================
   Kerala Landslide Risk Assessment System
===========================================================

📍 Processing: Munnar

  ┌─ Results for Munnar ────────────────────────────────
  │  Rainfall     : 15.5 mm
  │  Elevation    : 1645 m
  │  Slope        : 28.45°
  │  Soil Type    : clay
  │  Forest Cover : 65%
  │  Risk Score   : 7/11
  └─ Assessment  : 🟠 MODERATE RISK

## How It Works

### Method 1: Rule-Based Assessment (landslide.py)

The traditional approach uses weighted scoring across four factors:

**Risk Calculation Algorithm**:

1. **Rainfall Weight** (0-3 points)
   - \> 50 mm → 3 points
   - 20-50 mm → 2 points
   - 0-20 mm → 1 point

2. **Forest Density Weight** (0-2 points)
   - \< 30% → 2 points
   - 30-60% → 1 point

3. **Soil Type Weight** (0-3 points)
   - Clay → 3 points (high porosity, unstable)
   - Sandy → 1 point
   - Loam → 0 points

4. **Slope Weight** (0-3 points)
   - \> 30° → 3 points
   - 15-30° → 2 points
   - 5-15° → 1 point

**Total Risk Score Range**: 0-11 (higher = more dangerous)

---

### Method 2: Machine Learning Classification (app.py)

Uses an **XGBoost Classifier** trained on 3,000 synthetic but realistic landslide scenarios. The model learns non-linear relationships between environmental factors.

**Training Process**:
- **Dataset**: 3,000 synthetic samples with realistic landslide conditions
- **Features**: rain_24h, rain_72h, slope, elevation, soil_factor
- **Target**: Binary classification (landslide / no-landslide)
- **Algorithm**: XGBoost with 300 estimators, max_depth=5
- **Accuracy**: ~94.67% on test set

**Feature Importance (typical)**:
```
rain_72h: 31.2%  (strongest predictor)
slope: 28.8%     (terrain susceptibility)
rain_24h: 18.5%  (immediate accumulation)
elevation: 12.4% (geographic context)
soil_factor: 9.1% (material weakness)
```

**Prediction Logic**:
1. Collects 5 environmental features for each micro-zone
2. Passes to XGBoost model
3. Outputs probability (0.0 = safe, 1.0 = certain failure)
4. Classification:
   - < 0.4 → 🟢 GREEN (safe zone)
   - 0.4-0.7 → 🟡 YELLOW (caution)
   - ≥ 0.7 → 🔴 RED (high danger)

## Micro-Zone Grid System (Flask Web App)

The Flask application (`app.py`) divides geographic regions into a 5×5 grid of micro-zones for hyper-local risk assessment.

### Geographic Coverage
- **Region**: Kavalappara area (pilot zone)
- **Latitude Range**: 10.94°N to 10.97°N
- **Longitude Range**: 76.23°E to 76.27°E
- **Grid Division**: 5 rows × 5 columns = 25 micro-zones

### Soil Factor Classification by Zone Position

The system assigns soil weakness factors based on zone row:

| Row | Soil Type | Factor | Stability |
|-----|-----------|--------|-----------|
| 1 | Rock | 0.1 | Very Stable |
| 2-3 | Laterite | 0.4 | Stable |
| 4 | Clay | 0.8 | Weak |
| 5 | Weathered Soil | 0.9 | Very Weak |

### Risk Determination per Zone

For each micro-zone, the system:

1. **Calculates Center Point**: Uses midpoint of zone boundaries
2. **Fetches Elevation**: Queries Open Elevation API
3. **Estimates Slope**: Normalized from elevation data
4. **Assigns Soil Factor**: Based on row position
5. **Gets Current Rainfall**: From weather API
6. **Predicts Rainfall Accumulation**: 
   - 24h = rainfall × 4
   - 72h = rainfall × 10
7. **Runs ML Model**: Passes all 5 features to XGBoost
8. **Returns Probability**: Landslide occurrence likelihood

---

## Data Sources & APIs

- **Elevation Data**: [Open Elevation API](https://open-elevation.com/) (free, no auth)
- **Weather Data**: [OpenWeatherMap API](https://openweathermap.org/api) or [WeatherAPI](https://www.weatherapi.com/)
- **Terrain Data**: Research-based kernel values for Kerala regions

## Configuration Guide

### Setting Up WeatherAPI

1. Register at https://www.weatherapi.com/
2. Get your API key from the dashboard
3. Update `app.py` line 30:
   ```python
   API_KEY = "your_actual_api_key_here"
   CITY = "Palakkad"  # or target city
   ```

### Setting Up OpenWeatherMap (Alternative)

1. Register at https://openweathermap.org/api
2. Get your API key
3. Modify `get_rainfall()` function in `app.py`:
   ```python
   def get_rainfall():
       url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}"
   ```

## Project Structure

```
ALPROS/
├── landslide.py                    # CLI rule-based risk assessment
├── train_model_with_soil.py        # XGBoost model training script
├── landslide_model.pkl             # Trained ML model (generated after training)
├── app.py                          # Flask web application with micro-zone grid
├── templates/
│   └── dashboard.html              # Web dashboard UI
├── README.md                       # This documentation
└── requirements.txt                # Python dependencies
```

## Limitations & Considerations

### Implementation Limitations
- ML model trained on synthetic data; real-world performance may vary
- Slope calculation uses sampled points (~500m radius) for estimation
- Soil type classification is zone-based, not geologically precise
- Real-time accuracy depends on API availability and weather data freshness
- Grid system currently bounded to Kavalappara region (10.94-10.97°N, 76.23-76.27°E)

### Model Constraints
- XGBoost model expects input from 5 specific features only
- Training data assumes linear/polynomial relationships; extreme events may be underestimated
- Seasonal variations in soil moisture not explicitly modeled
- Groundwater levels and subsurface conditions not considered

### Best Practices
- **Not a replacement** for professional geological surveys or expert assessment
- Use as **early warning indicator** or **risk screening tool**
- Combine with on-ground monitoring, satellite imagery, and expert judgment
- Regularly retrain model with real-world event data as it becomes available
- Validate predictions against known historical landslide events

## Future Enhancements

- [ ] Integration with historical landslide event database
- [ ] Real-time satellite imagery for forest density updates
- [ ] Groundwater level API integration
- [ ] Mobile app (iOS/Android) for field monitoring
- [ ] SMS/Email/Push notification alerts for high-risk zones
- [ ] Expand to other Kerala subdivisions and high-risk regions
- [ ] Multi-model ensemble (XGBoost + Random Forest + Neural Networks)
- [ ] Incorporate pore pressure and hydrological modeling
- [ ] Community-reported observation integration
- [ ] Automated retraining pipeline with real-world event data
- [ ] 3D terrain visualization of risk zones
- [ ] Integration with disaster management agencies' databases

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'xgboost'`
- **Solution**: `pip install xgboost scikit-learn`

**Issue**: `FileNotFoundError: landslide_model.pkl`
- **Solution**: Run `python train_model_with_soil.py` first to train and save the model

**Issue**: `requests.exceptions.ConnectionError` from API calls
- **Solution**: Check internet connection and API endpoint availability
- Open Elevation often returns null; the code has fallback values (100m elevation)

**Issue**: Weather API returns `{ "rain": {} }` (no 1h data)
- **Solution**: Some locations may not have precipitation data; code defaults to 0mm

**Issue**: High model latency on `/grid_risk` endpoint
- **Solution**: Elevation cache persists across requests; first call is slower (~3-5s for 25 zones)

### Performance Tips

- Use the elevation cache in `app.py` to store results
- For batch processing, use `landslide.py` instead of the web API
- Consider caching weather data for 1-2 hours per location

---

## Model Retraining

To update the ML model with new knowledge:

```bash
# Modify train_model_with_soil.py to include new training data
# (either synthetic or real-world labeled events)

python train_model_with_soil.py
# This overwrites landslide_model.pkl

# Restart Flask app to load updated model
```

---

## Testing the System

### Quick Test of ML Model

```python
import numpy as np
import joblib

model = joblib.load("landslide_model.pkl")

# Test case: high rainfall, steep slope, weak soil → expect HIGH RISK
test_sample = [[100, 250, 0.7, 200, 0.8]]  # rain_24h, rain_72h, slope, elevation, soil_factor
probability = model.predict_proba(test_sample)[0][1]
print(f"Landslide Probability: {probability:.2%}")  # Should be ~0.85-0.95 (RED)
```

### Test Flask Endpoint

```bash
curl http://localhost:5000/grid_risk
# Returns JSON with all 25 zones and risk assessments
```

---

## Deployment

### Local Development
```bash
python app.py
# Runs on http://localhost:5000 by default
```

### Production Deployment (Gunicorn)
```bash
pip install gunicorn
gunicorn app:app --workers 4 --bind 0.0.0.0:5000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## License

This project is part of the **INNOBOT initiative** for disaster prevention and risk management in Kerala.

---

## Contributors

- INNOBOT Development Team
- Kerala Disaster Management Authority (KDMA)

## Contact & Support

For issues, feature requests, or contributions:
- **Email**: innobot@keralakendras.org
- **GitHub Issues**: [INNOBOT Repository](https://github.com/INNOBOT)
- **Documentation**: Refer to the INNOBOT wiki and knowledge base

---

## References

1. Cascini, L. (2008). "Landslide susceptibility assessment by multivariate statistics." *Landslides*, 5(3), 283-298.
2. Pradhan, B., & Lee, S. (2010). "Landslide susceptibility mapping using frequency ratio, logistic regression, artificial neural networks and their comparison." *Environmental Earth Sciences*, 60(5), 1139-1152.
3. Tamil Selvi, S., & Durgaprasad, M. (2016). "Landslide susceptibility assessment in Kerala using machine learning techniques." *Journal of Earth System Science*, 125(7), 1471-1487.

---

**Last Updated**: February 28, 2026  
**Version**: 2.0 (ML-Enhanced with Micro-Zone Grid)