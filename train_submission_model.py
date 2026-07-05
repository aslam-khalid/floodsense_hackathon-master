import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv("floodsense_training_data.csv")

# Clean data as per notebook
df = df[~df["district"].isin(["Nowshera", "Jacobabad"])]
sindh_kp_mean_precip = df[df["district"].isin(["Sindh_District", "KP_District"])]["precipitation"].mean()
df.loc[df["district"] == "Balochistan_District", "precipitation"] = sindh_kp_mean_precip
df = df[df["precipitation"] != -999]
df = df[df["soil_moisture"] != 5.0]
df = df[df["elevation"] != 99999]
df.drop_duplicates(inplace=True)

df["water_area_pct_change"] = df["water_area_pct_change"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)

for col in ["precipitation", "precip_3day_avg", "precip_7day_avg"]:
    df[col] = df[col].fillna(0)
    
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 2. Define features used by run_prediction.py
FEATURE_COLS = [
    "precipitation", "precip_3day_avg", "precip_7day_avg",
    "soil_moisture", "soil_3day_avg",
    "water_area_km2", "water_area_change", "water_area_pct_change",
    "temperature", "humidity", "pressure", "evaporation", "wind_speed",
    "month", "day_of_year", "is_monsoon", "ds_idx",
]

# Create X with all 17 features
X = df[FEATURE_COLS].copy()
y = df["flood_event"]

# Zero out the leakage features so the model ignores them
# but they still exist in the feature array to prevent mismatch errors.
LEAKAGE_COLS = ["water_area_change", "ds_idx"]
for col in LEAKAGE_COLS:
    X[col] = 0

# 3. SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train)

# 4. Train Model
xgb_model = xgb.XGBClassifier(
    n_estimators=383, 
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_s, y_train_s)

# 6. Save directly as model.pkl
joblib.dump(xgb_model, "model.pkl")
print("Raw XGBoost Model saved to model.pkl successfully.")
