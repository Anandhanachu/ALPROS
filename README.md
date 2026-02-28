# ALPROS - Automatic Lanscape Prediction & Rescue Optimiztion System

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
  - **GREEN**: Low risk (probability < 0.4)
  - **YELLOW**: Moderate risk (0.4 ≤ probability < 0.7)
  - **RED**: High risk (probability ≥ 0.7)

### Region-Specific Analysis
- **Predefined Regions**:
  - Kavalappara
  - Munnar
  - Wayanad
  - Malappuram
- Each region is defined by latitude/longitude bounding boxes and linked to the nearest OpenWeatherMap city for rainfall data.

### APIs
- **Grid Risk API**: `/grid_risk?region=<region>` - Returns risk assessment for all micro-zones in the specified region.
- **Regions API**: `/regions` - Lists all available regions with their display names and associated cities.

### Model Training
- **Synthetic Data Generation**: Simulates realistic rainfall, slope, elevation, and soil conditions.
- **Logical Landslide Condition**: Incorporates domain knowledge to define landslide-prone scenarios.
- **XGBoost Classifier**: Trained on 3,000 samples with a test accuracy of ~90%.
- **Feature Importance**:
  - Rainfall (72-hour) and slope are the most critical factors influencing landslide risk.

### Caching for Performance
- **Elevation Cache**: Stores elevation data for lat/lon coordinates to reduce API calls.
- **Rainfall Cache**: Caches rainfall data for cities to optimize OpenWeatherMap API usage.
- **Soil Factor Cache**: Caches soil classification data retrieved from Bhuvan WMS.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/alpros.git
   cd alpros
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenWeather

Map

 API key:
   - Replace `YOUR_OPENWEATHER_API_KEY` in `app.py` with your actual API key.

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Access the web dashboard at `http://127.0.0.1:5000/dashboard`.

## Usage

### Train the Model
To retrain the model, run:
```bash
python train_model_with_soil.py
```
This will generate a new [`landslide_model.pkl`]

### API Endpoints
- **Home**: [`http://127.0.0.1:5000/`]
- **Regions**: [`http://127.0.0.1:5000/regions`]
- **Grid Risk**: [`http://127.0.0.1:5000/grid_risk?region=<region>`]

## Future Enhancements
- Add support for additional regions.
- Integrate more advanced soil and geological data sources.
- Improve real-time data handling for rainfall and elevation.