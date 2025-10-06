# RDU Hourly Temperature Forecast (Two Weeks Ahead)

**Goal:** Predict hourly air temperature at **Raleigh–Durham International Airport (RDU)** for  
**00:00 Sep 17 – 23:00 Sep 30 (14 days)** under strict **no-leakage rules**.  
Each model must make all hourly predictions **before** the test window begins.


## 1. Project Overview
This project develops and evaluates machine learning models for short-term to mid-range temperature forecasting.  
We use only data **available prior to Sep 17 00:00**, ensuring a realistic “future prediction” setup.  
The pipeline includes feature construction, model training, offline evaluation, and a one-shot inference script that produces the final forecast CSV.


## 2. Key Rules
- Any data source is allowed **only if issued before the test window (≤ Sep 16 23:59)**.  
- You must include **one Linear Regression model** and **at least one other model**.  
- No actual observations from Sep 17–30 can be used as features.  
- All predictions must cover every hour of the test period once and only once.


## 3. Folder Structure
```
project-rdu-temp-forecast-14day/
├── data/
│   ├── raw/                         # Raw station observations up to Sep 16
│   ├── external/                    # Pre-issued forecasts or external data
│   └── processed/                   # Finalized training & inference features
│       ├── sample_train.csv
│       ├── sample_infer.csv
│       ├── train_full.csv
│       └── infer_features_2025-09-17_to_09-30.csv
│
├── src/
│   ├── features/
│   │   └── build_features.py        # Build features for training/inference
│   ├── models/
│   │   ├── lr_model.py              # Linear Regression baseline
│   │   ├── model_a.py               # Non-linear model A (e.g. XGBoost)
│   │   └── model_b.py               # Non-linear model B (optional)
│   ├── eval/
│   │   └── evaluate.py              # Evaluate models & produce plots
│   └── predict/
│       └── make_forecast.py         # Generate forecast for Sep 17–30
│
├── models/                          # Saved model artifacts (.pkl)
├── submissions/                     # Final forecast submissions
├── reports/                         # Figures / slides / writeup
├── FEATURE_SPEC.md                  # Feature definitions and units
├── DATA_SOURCES.md                  # Data provenance notes
├── requirements.txt                 # Dependencies
├── Makefile                         # Shortcut commands
└── README.md                        # Project documentation
```


## 4. No-Leakage Policy
1. **Training data** must end ≤ Sep 16 23:59 (local time or UTC — be consistent).  
2. **Inference features** for Sep 17–30 must be available at Day 0 (Sep 17 00:00).  
3. If additional weather variables are used, only their **forecasts issued before Sep 17** may appear.  
4. The code performs lightweight safety checks to ensure these constraints.


## 5. Quickstart
```bash
# 0) Environment setup
pip install -r requirements.txt

# 1) Build features
python -m src.features.build_features --mode train \
    --out data/processed/train_full.csv
python -m src.features.build_features --mode infer \
    --out data/processed/infer_features_2025-09-17_to_09-30.csv

# 2) Train models
python -m src.models.lr_model  --train data/processed/train_full.csv --save models/lr.pkl
python -m src.models.model_a   --train data/processed/train_full.csv --save models/model_a.pkl
python -m src.models.model_b   --train data/processed/train_full.csv --save models/model_b.pkl

# 3) Evaluate models (overall + by-horizon)
python -m src.eval.evaluate \
    --train data/processed/train_full.csv \
    --models models/lr.pkl models/model_a.pkl models/model_b.pkl \
    --out reports/figures

# 4) Make forecast (choose best model)
python -m src.predict.make_forecast \
    --features data/processed/infer_features_2025-09-17_to_09-30.csv \
    --model models/model_a.pkl \
    --out submissions/rdu_temp_2025-09-17_to_09-30.csv
```

## 6. Models
- **Linear Regression (Baseline):**  
  Uses regularized Ridge or Lasso regression with cyclical time features (e.g., sine/cosine encodings for hour and day-of-year), and lagged/rolling statistics (e.g., `temp_lag_24`, `roll_mean_24`).

- **Model A / Model B (Non-linear Models):**  
  Built using algorithms such as **XGBoost**, **LightGBM**, or **Random Forest**.  
  Optionally, a simple **LSTM or RNN** may be tested for sequential dependencies.

All models are trained using identical feature columns for fair comparison.  
Hyperparameters and feature selections can be tuned independently.  
The final report and presentation include both performance comparison and discussion of model limitations.


## 7. Evaluation
- **Metrics:**  
  - Mean Absolute Error (**MAE**)  
  - Root Mean Squared Error (**RMSE**)  
  - Horizon-based evaluation:  
    - **D1–2:** near-term (first 48 hours)  
    - **D3–7:** mid-term (next 5 days)  
    - **D8–14:** long-term (final week)

- **Visualizations:**  
  - Error vs. forecast-horizon curve  
  - Model comparison bar charts  
  - Optional residual plots or scatter plots  

All evaluation artifacts are saved under `reports/figures/`,  
including plots and a summary metrics table (`metrics.csv`).


## 8. Forecast Output Format
The final forecast is saved as:

**File:** `submissions/rdu_temp_2025-09-17_to_09-30.csv`

**Structure:**

- One row per hour → **336 rows total** (14 days × 24 hours).  
- Columns:
  - `timestamp` – ISO 8601 format (`YYYY-MM-DDTHH:MM:SS`)  
  - `temp_pred` – predicted temperature value  
- Temperature unit and timezone are detailed in `FEATURE_SPEC.md`.


## 9. License
This project is released under the **MIT License**.  
You are free to use, modify, and share it for educational or research purposes.

---

*Clean, reproducible, and leakage-free forecasting —  
a professional-grade ML workflow for time-series prediction.*
