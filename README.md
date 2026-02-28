# ALPROS - Kerala Landslide Risk Assessment System

ALPROS is an intelligent landslide risk prediction system designed to assess and monitor landslide hazards across Kerala's vulnerable regions. It leverages real-time weather data and terrain analysis to provide accurate risk assessments.

## Features

- **Real-time Weather Integration**: Fetches current precipitation data via WeatherAPI
- **Elevation & Slope Analysis**: Calculates terrain slope using Open Elevation API
- **Multi-factor Risk Assessment**: Evaluates risk based on:
  - Current rainfall conditions
  - Terrain slope angle
  - Soil composition
  - Forest density coverage
  - Elevation data
- **Intelligent Risk Scoring**: Generates risk scores (0-11) with visual indicators
  - 🔴 **HIGH RISK** (score ≥ 8)
  - 🟠 **MODERATE RISK** (score 5-7)
  - 🟢 **LOW RISK** (score < 5)

## Supported Locations

Currently monitors the following Kerala districts:
- **Munnar** - High forest density, clay soil, mountainous terrain
- **Wayanad** - Dense forests, clay soil, elevated region
- **Idukki** - Moderate forest cover, clay soil, hilly terrain
- **Kochi** - Low forest density, sandy soil, coastal area
- **Kottayam** - Medium forest coverage, loam soil

## Prerequisites

- Python 3.7 or higher
- `requests` library
- Internet connection (for API calls)

### Installation

```bash
pip install requests
```

## Configuration

The system requires two API keys:

1. **WeatherAPI Key**: Register at [weatherapi.com](https://www.weatherapi.com/)
   - Free tier available with 1 million calls/month
   - Update `WEATHER_API_KEY` in `landslide.py`

2. **Open Elevation API**: Free, no authentication required
   - Fetches global elevation data

## Usage

Run the assessment system:

```bash
python landslide.py
```

### Output Example

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
```

## How It Works

### Risk Calculation Algorithm

The system evaluates four primary risk factors:

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

## Data Sources

- **Elevation Data**: [Open Elevation API](https://open-elevation.com/)
- **Weather Data**: [WeatherAPI](https://www.weatherapi.com/)
- **Terrain Data**: Research-based values for Kerala districts

## Project Structure

```
ALPROS/
├── landslide.py      # Main assessment engine
└── README.md         # Documentation
```

## Limitations

- Slope calculation uses sampled points (~500m radius) for estimation
- Soil type and forest density are static values based on district averages
- Real-time accuracy depends on API availability and data refresh rates
- Ideal for general risk indication, not a substitute for professional geological surveys

## Future Enhancements

- [ ] Historical landslide data integration
- [ ] Machine learning-based risk prediction
- [ ] Mobile app interface
- [ ] SMS/Email alerts for high-risk zones
- [ ] Integration with satellite imagery for forest density updates
- [ ] Support for more Kerala subdivisions

## License

This project is part of the INNOBOT initiative for disaster prevention and risk management in Kerala.

## Support

For issues, improvements, or contributions, please refer to the INNOBOT documentation.