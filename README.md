# RDU 14-Day Temperature Forecast

Predict hourly temperature at Raleigh-Durham Airport (RDU) for **Sep 17-30, 2025** (336 hours) using machine learning models trained on historical data.

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Fetch training data (2020-2025)
python src/data/api_xgb_data.py

# Build features
python src/data/build_features.py

# Open notebooks to train and evaluate models
jupyter notebook
```

## Project Structure

```
rdu-temp-forecast-14day/
├── data/
│   ├── raw/              # Raw API data
│   ├── staging/          # Standardized data
│   ├── processed/        # Feature-engineered datasets
│   └── results/          # Model predictions
├── src/data/             # Data fetching & feature engineering
├── models/               # Saved models (.pkl)
├── linear_regression_model.ipynb
├── xgboost_model.ipynb
├── RandomForest.ipynb
└── LSTM.ipynb
```

## Models Implemented

| Model | Test MAE | Test RMSE | Test R² | Status |
|-------|----------|-----------|---------|--------|
| **Linear Regression** | 0.56°C | 0.70°C | 0.97 | ✅ Best for deployment |
| **XGBoost** | 3.09°C | 3.80°C | 0.08 | ⚠️ Lag dependency issue |
| **LSTM** | 2.83°C | 3.64°C | 0.85 | ✅ Works but less accurate |
| **Random Forest** | 4.01°C | 4.72°C | -0.42 | ✅ Works but less accurate |

**Note:** XGBoost achieves 0.44°C MAE on validation but degrades to 3.09°C on realistic test due to unavailable lag features during true forecasting.

## Key Features

- **Time features:** Hour (sin/cos), day of year (sin/cos), day of week, weekend
- **Temperature lags:** 1h, 24h, 48h, 72h, 168h (past observations)
- **Rolling statistics:** Mean and std over 24h and 168h windows
- **Weather features:** Humidity, pressure, dew point (Linear Regression only)

## Data Source

- **API:** [Open-Meteo](https://open-meteo.com/) historical weather archive
- **Location:** RDU Airport (35.8776°N, 78.7875°W)
- **Training period:** Jan 2020 - Sep 16, 2025
- **Test period:** Sep 17-30, 2025 (336 hours)

## No-Leakage Rules

1. Training data ends at **Sep 16, 2025 23:00**
2. Test data (Sep 17-30) is never used for training
3. Temporal split: earlier data for training, later for validation
4. No shuffling to simulate real forecasting conditions

## Critical Finding

**Lag Feature Dependency Issue:**
- XGBoost relies heavily on recent temperature observations (e.g., `temp_lag_1`, `temp_lag_24`)
- In real deployment, these features are unavailable for the 14-day forecast window
- Using historical climatology as a proxy degrades performance significantly (0.44°C → 3.09°C MAE)
- **Linear Regression** doesn't use lag features and is more robust for deployment

## Running the Models

Open each notebook in Jupyter and run all cells:
- `linear_regression_model.ipynb` - Best for real forecasting
- `xgboost_model.ipynb` - Best validation performance (if lags available)
- `LSTM.ipynb` - Deep learning approach
- `RandomForest.ipynb` - Needs improvement

## Citations:
GPT4o, Claude Sonnet 4 models were used to assist with ideation and code help + refinement.

## License

MIT License - Free to use for educational and research purposes.
