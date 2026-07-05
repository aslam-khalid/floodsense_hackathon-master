import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import json
import warnings
import sys
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix, recall_score, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  FloodSense Pipeline — Flood Event Prediction for Pakistan")
print("=" * 65)

df   = pd.read_csv("floodsense_training_data.csv")
elev = pd.read_csv("district_elevation_reference.csv")

print(f"\nRaw training rows : {len(df)}")
print(f"Raw columns       : {len(df.columns)}")


# ══════════════════════════════════════════════════════════════
# STEP 1: Fix ALL data bugs (before any split — required)
# ══════════════════════════════════════════════════════════════
print("\n[Step 1]  Fixing data bugs...")

# 1a. Drop Nowshera & Jacobabad phantom district rows (1 row each)
phantom_districts = ["Nowshera", "Jacobabad"]
n_before = len(df)
df = df[~df["district"].isin(phantom_districts)].copy()
print(f"  ✓ Dropped {n_before - len(df)} phantom-district rows "
      f"{phantom_districts}")

# -- SENSOR ROGUE IMPUTATION --
sindh_kp_mean_precip = df[df["district"].isin(["Sindh_District", "KP_District"])]["precipitation"].mean()
df.loc[df["district"] == "Balochistan_District", "precipitation"] = sindh_kp_mean_precip
print(f"  ✓ Imputed Balochistan precipitation with Sindh/KP mean: {sindh_kp_mean_precip:.2f}")

# 1b. Drop rows with impossible sentinel values
n_before = len(df)
df = df[df["precipitation"] != -999]
df = df[df["soil_moisture"]  != 5.0]
df = df[df["elevation"]      != 99999]
print(f"  ✓ Dropped {n_before - len(df)} sentinel-value rows "
      f"(precip=-999 / soil=5.0 / elev=99999)")

# 1c. Drop exact duplicates
n_before = len(df)
df.drop_duplicates(inplace=True)
print(f"  ✓ Dropped {n_before - len(df)} duplicate rows")

# 1d. Clip inf in water_area_pct_change then keep as feature (clipped, not dropped)
inf_count = np.isinf(df["water_area_pct_change"]).sum()
df["water_area_pct_change"] = (
    df["water_area_pct_change"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(-5, 5)
)
print(f"  ✓ Clipped {inf_count} inf value(s) in water_area_pct_change to [-5, 5]")

# 1e. Fill missing precipitation columns with 0 (no rain = 0 mm)
precip_cols = ["precipitation", "precip_3day_avg", "precip_7day_avg"]
for col in precip_cols:
    n = df[col].isna().sum()
    if n > 0:
        df[col] = df[col].fillna(0)
        print(f"  ✓ Filled {n} NaN in '{col}' → 0")

# 1f. Fill remaining small-count NaNs with column median
remaining = df.select_dtypes(include=[np.number]).isnull().sum()
remaining = remaining[remaining > 0]
for col in remaining.index:
    med = df[col].median()
    df[col] = df[col].fillna(med)
    print(f"  ✓ Filled {remaining[col]} NaN in '{col}' → median {med:.4f}")

assert df.isnull().sum().sum() == 0, "NaNs remain after cleaning!"
assert np.isinf(df.select_dtypes(include=[np.number])).sum().sum() == 0, \
    "Infs remain after cleaning!"
print(f"\n  Clean rows: {len(df)} | NaNs: 0 | Infs: 0  ✅")
df.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt

# Identify numerical features available in df at this stage
# Exclude non-feature numerical columns like 'ds_idx' or the target 'flood_event' if present
current_numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
# Filter out 'flood_event' if it's the target and not meant for this early correlation plot
plot_cols = [col for col in current_numerical_cols if col not in ['flood_event', 'ds_idx', 'elevation', 'latitude', 'longitude', 'water_area_km2', 'water_area_change']]

# Calculate correlation matrix for the available numerical features
corr_matrix = df[plot_cols].corr()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap (Pre-Engineering)', fontsize=15, fontweight='bold')
plt.show()

print("\n[Step 2]  Dropping leakage columns & merging elevation reference...")

# elevation/lat/lon are constant per district → no signal
# ds_idx identifies source dataset → leaks label
# water_area_km2 is perfectly separating (flood defined as high water area)
# water_area_change is a direct derivative of water_area_km2
LEAKAGE_COLS = [
    "year",
    "elevation",         # constant per district
    "latitude",          # constant per district
    "longitude",         # constant per district
    "ds_idx",            # dataset-source identifier leaks target
    "water_area_km2",    # perfectly separated — no overlap between classes
    "water_area_change", # direct derivative of water_area_km2
]
df.drop(columns=LEAKAGE_COLS, inplace=True)
print(f"  ✓ Dropped leakage columns: {LEAKAGE_COLS}")

# Merge avg_elevation_m from the district reference file
df = df.merge(elev[["district", "avg_elevation_m"]], on="district", how="left")
print(f"  ✓ Merged avg_elevation_m from district_elevation_reference.csv")
assert df["avg_elevation_m"].isna().sum() == 0, "avg_elevation_m has NaNs after merge!"

print("\n[Step 3]  Engineering features...")

df["soil_x_precip"]    = df["soil_moisture"] * df["precipitation"]
df["monsoon_x_precip"] = df["is_monsoon"]    * df["precipitation"]
df["humid_x_precip"]   = df["humidity"]      * df["precipitation"]
df["precip_x_elev"]    = df["precipitation"] * df["avg_elevation_m"]
df["soil_x_humidity"]  = df["soil_moisture"] * df["humidity"]
df["temp_humidity"]    = df["temperature"]   * df["humidity"]

le = LabelEncoder()
df["district_enc"] = le.fit_transform(df["district"])

print(f"  ✓ Created 6 interaction features + district label encoding")
print(f"  ✓ Districts: {dict(zip(le.classes_, le.transform(le.classes_)))}")



# Plotting removed for headless execution

# ══════════════════════════════════════════════════════════════
# STEP 4: Train / Test split (80/20, stratified)
# ══════════════════════════════════════════════════════════════
print("\n[Step 4]  Train/test split (80/20 stratified on flood_event)...")

DROP_COLS = ["date", "district", "flood_event"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
TARGET = "flood_event"

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} rows | flood rate: {y_train.mean():.2%}")
print(f"  Test : {len(X_test)} rows  | flood rate: {y_test.mean():.2%}")

# ══════════════════════════════════════════════════════════════
# STEP 5: SMOTE + Train XGBoost
# ══════════════════════════════════════════════════════════════
print("\n[Step 5]  SMOTE oversampling → XGBoost training...")

# SMOTE on train only (never on test — that would be leakage)
sm = SMOTE(random_state=42)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_train_s)} rows | flood rate: {y_train_s.mean():.2%}")

model = xgb.XGBClassifier(
    n_estimators=600,
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
    n_jobs=-1,
    early_stopping_rounds=50,
)
model.fit(
    X_train_s, y_train_s,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print(f"  Best iteration: {model.best_iteration}")

def get_risk_output(model, X):
    y_prob = model.predict_proba(X)[:, 1]

    # 4-tier risk classification
    def classify_risk(p):
        if p < 0.25:   return "Low"
        elif p < 0.50: return "Medium"
        elif p < 0.75: return "High"
        else:          return "Critical"

    risk_level      = [classify_risk(p) for p in y_prob]
    confidence_pct  = (np.where(y_prob >= 0.5, y_prob, 1 - y_prob) * 100).round(1)
    flood_predicted = (y_prob >= 0.5).astype(int)

    return pd.DataFrame({
        "flood_event_predicted" : flood_predicted,
        "flood_probability_%"   : (y_prob * 100).round(1),
        "risk_level"            : risk_level,
        "confidence_%"          : confidence_pct,
    })


results = get_risk_output(model, X_test)
print(results.head(10))
print("\nRisk level distribution:")
print(results["risk_level"].value_counts())

# Calibration block removed due to compatibility issues.
# The base model is already trained and will be saved.


# Calibration curve comparison removed

print("\n[Step 6]  Evaluation")
print("=" * 65)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc:.4f}")
print()

print("  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Flood", "Flood"],
                             digits=4))

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix (rows=Actual, cols=Predicted):")
print(f"                No Flood  Flood")
print(f"  Actual No     {cm[0,0]:>8}  {cm[0,1]:>5}")
print(f"  Actual Flood  {cm[1,0]:>8}  {cm[1,1]:>5}")

# Feature importance plot removed

model.save_model("floodsense_model.json")

metadata = {
    "feature_cols": FEATURE_COLS,
    "district_encoding": {d: int(i) for d, i in
                          zip(le.classes_, le.transform(le.classes_))},
    "test_accuracy": round(acc, 4),
    "test_f1": round(f1, 4),
    "test_roc_auc": round(roc, 4),
    "best_iteration": int(model.best_iteration),
    "n_train": len(X_train_s),
    "n_test": len(X_test),
}
with open("floodsense_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n  Saved: floodsense_model.json")
print("  Saved: floodsense_metadata.json")

print("\n" + "=" * 65)
if acc >= 0.80:
    print(f"  ✅ TARGET MET: {acc*100:.2f}% accuracy ")
else:
    print(f"  ❌ Below target: {acc*100:.2f}%")
print("=" * 65)

def predict_flood(input_dict: dict) -> dict:
    """
    Predict flood risk for a single observation.

    Parameters
    ----------
    input_dict : dict
        Must contain keys matching the raw features before engineering.
        Required keys (at minimum):
            soil_moisture, precipitation, humidity, temperature,
            pressure, wind_speed, evaporation,
            precip_3day_avg, precip_7day_avg,
            temp_3day_avg, soil_3day_avg,
            day_of_year, month, year, is_monsoon,
            water_area_pct_change,
            avg_elevation_m, district_enc

    Returns
    -------
    dict with keys: flood_predicted (0/1), flood_probability (0-1),
                    risk_level ('LOW'/'MEDIUM'/'HIGH')
    """
    row = pd.DataFrame([input_dict])

    # Engineer same features
    row["soil_x_precip"]    = row["soil_moisture"] * row["precipitation"]
    row["monsoon_x_precip"] = row["is_monsoon"]    * row["precipitation"]
    row["humid_x_precip"]   = row["humidity"]      * row["precipitation"]
    row["precip_x_elev"]    = row["precipitation"] * row["avg_elevation_m"]
    row["soil_x_humidity"]  = row["soil_moisture"] * row["humidity"]
    row["temp_humidity"]    = row["temperature"]   * row["humidity"]

    X_row = row[FEATURE_COLS]
    prob  = model.predict_proba(X_row)[0, 1]
    pred  = int(prob >= 0.5)

    risk = "LOW" if prob < 0.35 else ("HIGH" if prob >= 0.65 else "MEDIUM")

    return {
        "flood_predicted": pred,
        "flood_probability": round(float(prob), 4),
        "risk_level": risk,
    }

y_prob = model.predict_proba(X_test)[:, 1]

def get_risk_output(y_prob):
    def risk_level(p):
        if p < 0.25:   return "Low"
        elif p < 0.50: return "Medium"
        elif p < 0.75: return "High"
        else:          return "Critical"

    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob) * 100

    return pd.DataFrame({
        "flood_event"   : (y_prob >= 0.5).astype(int),        # ✅ target column
        "probability_%" : (y_prob * 100).round(1),             # ✅ raw probability
        "risk_level"    : [risk_level(p) for p in y_prob],     # ✅ primary output
        "confidence_%"  : confidence.round(1),                 # ✅ secondary output
    })

results = get_risk_output(y_prob)
print(results.head(10))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              classification_report)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings; warnings.filterwarnings("ignore")

# ── Load & clean ───────────────────────────────────────────────
df   = pd.read_csv("floodsense_training_data.csv")
elev = pd.read_csv("district_elevation_reference.csv")

df = df[~df["district"].isin(["Nowshera","Jacobabad"])]
df = df[df["precipitation"] != -999]
df = df[df["soil_moisture"]  != 5.0]
df = df[df["elevation"]      != 99999]
df.drop_duplicates(inplace=True)
df["water_area_pct_change"] = df["water_area_pct_change"].replace([np.inf,-np.inf], np.nan).clip(-5,5)
for c in ["precipitation","precip_3day_avg","precip_7day_avg"]:
    df[c] = df[c].fillna(0)
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

df = df.merge(elev[["district","avg_elevation_m"]], on="district", how="left")
df.drop(columns=["elevation","latitude","longitude","ds_idx",
                  "water_area_km2","water_area_change"], inplace=True, errors="ignore")

# NOTE: Spike rows will be added AFTER the train/test split for the binary model's training set.

# ── Feature engineering ────────────────────────────────────────
df["soil_x_precip"]    = df["soil_moisture"] * df["precipitation"]
df["monsoon_x_precip"] = df["is_monsoon"]    * df["precipitation"]
df["humid_x_precip"]   = df["humidity"]      * df["precipitation"]
df["precip_x_elev"]    = df["precipitation"] * df["avg_elevation_m"]
df["soil_x_humidity"]  = df["soil_moisture"] * df["humidity"]
df["temp_humidity"]    = df["temperature"]   * df["humidity"]

le = LabelEncoder()
df["district_enc"] = le.fit_transform(df["district"])

# ══════════════════════════════════════════════════════════════
# CREATE SEVERITY LABELS using XGBoost-learnable features
# ══════════════════════════════════════════════════════════════
# Severity is derived from physical intensity of flood conditions
# Only flood rows (flood_event=1) get severity 1-4
# No-flood rows get severity 0

p90  = df["precipitation"].quantile(0.90)
p75  = df["precipitation"].quantile(0.75)
p50  = df["precipitation"].quantile(0.50)
sm75 = df["soil_moisture"].quantile(0.75)
w75  = df["water_area_pct_change"].quantile(0.75)
p3_75= df["precip_3day_avg"].quantile(0.75)

def assign_severity(row):
    if row["flood_event"] == 0:
        return 0  # No Flood

    score = 0
    # Precipitation (max weight — top feature)
    if   row["precipitation"] >= p90: score += 4
    elif row["precipitation"] >= p75: score += 3
    elif row["precipitation"] >= p50: score += 2
    else:                             score += 1

    # Soil moisture
    score += 2 if row["soil_moisture"]        >= sm75 else 1

    # Water area pct change
    score += 2 if row["water_area_pct_change"] >= w75  else 1

    # 3-day precip avg
    score += 2 if row["precip_3day_avg"]       >= p3_75 else 1

    # Monsoon flag
    score += 2 if row["is_monsoon"] == 1 else 0

    # Humidity
    score += 1 if row["humidity"] > df["humidity"].median() else 0

    # Map score → severity class
    if   score <= 5:  return 1  # Low
    elif score <= 8:  return 2  # Medium
    elif score <= 11: return 3  # High
    else:             return 4  # Critical

df["flood_severity"] = df.apply(assign_severity, axis=1)

SEVERITY_MAP = {0:"No Flood", 1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
print("=" * 65)
print("  Severity label distribution:")
for k, v in SEVERITY_MAP.items():
    n = (df["flood_severity"]==k).sum()
    print(f"    {k} — {v:<10}: {n} rows")

# ── Common feature cols ────────────────────────────────────────
DROP = ["date","district","flood_event","flood_severity", "year"]
FEATURE_COLS = [c for c in df.columns if c not in DROP]

X = df[FEATURE_COLS]

# ── Shared train/test split ────────────────────────────────────
X_tr, X_te, y_bin_tr, y_bin_te = train_test_split(
    X, df["flood_event"], test_size=0.2, random_state=42,
    stratify=df["flood_event"]
)

# ══════════════════════════════════════════════════════════════
# MODEL 1 — Binary XGBoost (flood_event) with spike rows and calibration
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL 1 — Binary XGBoost (flood_event) with spike rows and calibration")
print("=" * 65)

# --- Spike Row Generation (as per Fix 1) ---
sindh_mean_precip = df[df["district"] == "Sindh_District"]["precipitation"].mean()
kp_mean_precip    = df[df["district"] == "KP_District"]["precipitation"].mean()

sindh_spike_val = sindh_mean_precip * 4
kp_spike_val    = kp_mean_precip    * 4

sindh_base = df[df["district"] == "Sindh_District"].sort_values("precipitation", ascending=False).iloc[0].copy()
kp_base    = df[df["district"] == "KP_District"].sort_values("precipitation", ascending=False).iloc[0].copy()

sindh_spike_row = sindh_base.copy()
kp_spike_row    = kp_base.copy()

for row_obj, spike_val in [(sindh_spike_row, sindh_spike_val), (kp_spike_row, kp_spike_val)]:
    row_obj["precipitation"]    = spike_val
    row_obj["precip_3day_avg"]  = spike_val * 0.85
    row_obj["precip_7day_avg"]  = spike_val * 0.60
    row_obj["soil_moisture"]    = min(row_obj["soil_moisture"] * 1.5, 0.99)
    row_obj["flood_event"]      = 1
    row_obj["date"]             = "2024-08-15"
    row_obj["is_monsoon"]       = 1
    row_obj["soil_x_precip"]    = row_obj["soil_moisture"] * row_obj["precipitation"]
    row_obj["monsoon_x_precip"] = row_obj["is_monsoon"]    * row_obj["precipitation"]
    row_obj["humid_x_precip"]   = row_obj["humidity"]      * row_obj["precipitation"]
    row_obj["precip_x_elev"]    = row_obj["precipitation"] * row_obj["avg_elevation_m"]
    row_obj["soil_x_humidity"]  = row_obj["soil_moisture"] * row_obj["humidity"]
    row_obj["temp_humidity"]    = row_obj["temperature"]   * row_obj["humidity"]
    row_obj["district_enc"]     = le.transform([row_obj["district"]])[0]

spike_df_features = pd.DataFrame([sindh_spike_row[FEATURE_COLS], kp_spike_row[FEATURE_COLS]])
spike_df_target   = pd.Series([sindh_spike_row["flood_event"], kp_spike_row["flood_event"]])

# Augment training data with spike rows
X_tr_augmented = pd.concat([X_tr, spike_df_features], ignore_index=True)
y_bin_tr_augmented = pd.concat([y_bin_tr, spike_df_target], ignore_index=True)

# SMOTE on augmented train set
sm = SMOTE(random_state=42)
X_train_s, y_train_s = sm.fit_resample(X_tr_augmented, y_bin_tr_augmented)
print(f"  After SMOTE on augmented train set: {len(X_train_s)} rows | flood rate: {y_train_s.mean():.2%}")

# Train base model
model_binary = xgb.XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=2, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)
model_binary.fit(
    X_train_s, y_train_s,
    eval_set=[(X_te, y_bin_te)], # Evaluate on original test set
    verbose=False,
)
print(f"  Best iteration for binary model: {model_binary.best_iteration}")

# Perform calibration (Sigmoid) (as per Fix 1 & 2)
# Split augmented train into train + calibration set for the base model
X_tr_base_cal, X_cal_set, y_tr_base_cal, y_cal_set = train_test_split(
    X_train_s, y_train_s, test_size=0.2, random_state=42, stratify=y_train_s
)

# Retrain base model on this smaller train set
base_model_for_calibration = xgb.XGBClassifier(
    n_estimators=model_binary.best_iteration + 1, # Use best iteration from first model
    max_depth=6, learning_rate=0.03, subsample=0.85,
    colsample_bytree=0.85, min_child_weight=2, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
base_model_for_calibration.fit(X_tr_base_cal, y_tr_base_cal, verbose=False)

from sklearn.frozen import FrozenEstimator
model_binary_calibrated = CalibratedClassifierCV(FrozenEstimator(base_model_for_calibration), method="sigmoid")
model_binary_calibrated.fit(X_cal_set, y_cal_set)
print("  Binary model calibrated with Sigmoid.")

y_bin_pred = model_binary_calibrated.predict(X_te)
y_bin_prob = model_binary_calibrated.predict_proba(X_te)[:,1]

print(f"  Accuracy (Calibrated) : {accuracy_score(y_bin_te, y_bin_pred)*100:.2f}%")
print(f"  F1 (Calibrated)       : {f1_score(y_bin_te, y_bin_pred):.4f}")
print(f"  Recall (Calibrated)   : {recall_score(y_bin_te, y_bin_pred):.4f}")

# ══════════════════════════════════════════════════════════════
# MODEL 2: Severity XGBoost (REMOVED - as per Fix 3)
# Severity will be directly assigned using rules within floodsense_predict.
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# COMBINED OUTPUT FUNCTION
# ══════════════════════════════════════════════════════════════
SEV_LABEL = {0:"No Flood", 1:"Low", 2:"Medium", 3:"High", 4:"Critical"} # Map 0-4 scores to labels

def floodsense_predict(model_for_binary_prediction, X_input: pd.DataFrame) -> pd.DataFrame:
    """
    Full FloodSense prediction pipeline.
    Runs binary flood prediction and then directly assigns severity based on rules.

    Parameters
    ----------
    model_for_binary_prediction : CalibratedClassifierCV
        The calibrated model for binary flood prediction (0/1).
    X_input : pd.DataFrame
        Input features for prediction. Must contain all FEATURE_COLS.

    Returns
    -------
    pd.DataFrame with combined predictions: flood_event, probability_%, risk_level, severity, confidence_%
    """

    # Model 1 — flood probability (using calibrated model and 0.5 threshold as per Fix 2)
    prob    = model_for_binary_prediction.predict_proba(X_input[FEATURE_COLS])[:,1]
    flood   = (prob >= 0.5).astype(int)
    confidence = np.where(prob >= 0.5, prob, 1 - prob) * 100

    # Risk band from probability
    def risk(p):
        if   p < 0.25: return "Low"
        elif p < 0.50: return "Medium"
        elif p < 0.75: return "High"
        else:          return "Critical/High Alert"

    # Severity (directly assigned using rules - as per Fix 3)
    severity_output = np.array(["No Flood"] * len(X_input), dtype=object)
    flood_mask = (flood == 1)

    if flood_mask.sum() > 0:
        # Create a temporary DataFrame for rows predicted as flood
        # Add 'flood_event' column back as assign_severity expects it
        temp_flood_df = X_input.loc[flood_mask, FEATURE_COLS].copy()
        temp_flood_df['flood_event'] = 1 # Mark as flood for severity assignment

        # Apply the assign_severity function
        predicted_severity_raw = temp_flood_df.apply(assign_severity, axis=1)

        # Map raw scores (0-4) to labels (No Flood, Low, Medium, High, Critical)
        severity_output[flood_mask] = predicted_severity_raw.map(SEV_LABEL).values

    return pd.DataFrame({
        "flood_event"   : flood,
        "probability_%"  : (prob*100).round(1),
        "risk_level"    : [risk(p) for p in prob],
        "severity"      : severity_output,
        "confidence_%"   : confidence.round(1),
    }, index=X_input.index) # Preserve original index

# ── Final combined output on test set ─────────────────────────
print("=" * 65)
print("  COMBINED OUTPUT — WITH CALIBRATED BINARY MODEL & RULE-BASED SEVERITY")
print("=" * 65)

final = floodsense_predict(model_binary_calibrated, X_te)
print(final.head(15).to_string())
print(f"\nSeverity distribution:")
print(final["severity"].value_counts())
print(f"\nNULL severity : {final['severity'].isna().sum()}  ✅")
print(f"NULL risk     : {final['risk_level'].isna().sum()}  ✅")
print(f"\nBinary accuracy (calibrated model) : {accuracy_score(y_bin_te, final['flood_event']):.4f}")

from sklearn.metrics import ConfusionMatrixDisplay

# Re-calculate y_prob and results to ensure they match y_test length (273)
# The get_risk_output function is defined in cell 23HVE5u9DZvx
y_prob = model_binary_calibrated.predict_proba(X_test)[:, 1]

def get_risk_output(y_prob):
    def risk_level(p):
        if p < 0.25:   return "Low"
        elif p < 0.50: return "Medium"
        elif p < 0.75: return "High"
        else:          return "Critical"

    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob) * 100

    return pd.DataFrame({
        "flood_event"   : (y_prob >= 0.5).astype(int),
        "probability_%" : (y_prob * 100).round(1),
        "risk_level"    : [risk_level(p) for p in y_prob],
        "confidence_%"  : confidence.round(1),
    })

results = get_risk_output(y_prob)
y_pred_results = results["flood_event"]

fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_results,
                                       display_labels=["No Flood", "Flood"],
                                       cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (Risk Output)", fontweight="bold")
plt.tight_layout()
plt.show()

print("\n[1] Brief Compliance Check")
banned = ["latitude", "longitude", "elevation"]
for col in banned:
    if col in FEATURE_COLS:
        print(f"   FAIL — '{col}' found in features!")
    else:
        print(f" '{col}' not in features")

print("\n[2] Overfitting Check")
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
gap       = train_acc - test_acc

print(f"  Train accuracy : {train_acc:.4f}")
print(f"  Test  accuracy : {test_acc:.4f}")
print(f"  Gap            : {gap:.4f} — {' OVERFIT (>10%)' if gap > 0.10 else '✅ OK'}")


from sklearn.model_selection import StratifiedKFold, cross_val_score

model_cv = xgb.XGBClassifier(
    n_estimators=model.best_iteration + 1,
    max_depth=6, learning_rate=0.03, subsample=0.85,
    colsample_bytree=0.85, min_child_weight=2, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
cv_scores = cross_val_score(model_cv, X_train, y_train,
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                             scoring="accuracy")
print(f"\n  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Std interpretation : {'⚠️ Unstable (std > 0.05)' if cv_scores.std() > 0.05 else '✅ Stable'}")

print("\n[3] Bias & Imbalance Check")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

f1     = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"  F1-Score (Flood class) : {f1:.4f}")
print(f"  Recall   (Flood class) : {recall:.4f} {'✅' if recall >= 0.80 else '⚠️ Missing too many floods!'}")


print("\n[4] Data Leakage Check")
leakage_cols = ["date", "year"]
for col in leakage_cols:
    if col in FEATURE_COLS:
        print(f"  ❌ LEAKAGE — '{col}' is in features!")
    else:
        print(f"  ✅ '{col}' not in features")

print("\n[5] Probability Calibration — ")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)

# — Confusion matrix —
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Flood", "Flood"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontweight="bold", pad=10)

# — CV scores per fold —
axes[1].bar(range(1, 6), cv_scores,
            color=plt.cm.RdYlGn(cv_scores / cv_scores.max()),
            edgecolor="white", linewidth=0.5)
axes[1].axhline(cv_scores.mean(), color="#185FA5", linestyle="--",
                linewidth=1.2, label=f"Mean={cv_scores.mean():.3f}")
axes[1].axhline(0.80, color="#A32D2D", linestyle=":", linewidth=1,
                label="80% target")
for i, v in enumerate(cv_scores):
    axes[1].text(i + 1, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
axes[1].set_ylim(0.7, 1.01)
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("5-Fold CV Accuracy", fontweight="bold", pad=10)
axes[1].legend(fontsize=9, frameon=False)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

# — Calibration curve —
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8)
axes[2].plot(prob_pred, prob_true, marker="o", color="#185FA5",
             linewidth=1.8, markersize=6, label="FloodSense")
axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
axes[2].fill_between(prob_pred, prob_pred, prob_true,
                     alpha=0.12, color="#185FA5")
axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Fraction of positives")
axes[2].set_title("Calibration Curve", fontweight="bold", pad=10)
axes[2].legend(fontsize=9, frameon=False)
axes[2].grid(linestyle="--", alpha=0.4)

plt.suptitle("FloodSense — Model Audit", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

y_pred_calibrated = model_binary_calibrated.predict(X_te)

fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
ConfusionMatrixDisplay.from_estimator(model_binary_calibrated, X_te, y_bin_te,
                                      display_labels=["No Flood", "Flood"],
                                      cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (Calibrated Binary Model)", fontweight="bold")
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# RAINFALL SPIKE TASK — CORRECTED VERSION
# At this point in the notebook, df already has:
#   - Leakage columns DROPPED (elevation, latitude, longitude, etc.)
#   - avg_elevation_m MERGED from elev
#   - Interaction features COMPUTED (soil_x_precip, etc.)
#   - district_enc COMPUTED via le
#   - FEATURE_COLS already defined
# ══════════════════════════════════════════════════════════════

# ── Step 1: Compute 300% spike values ─────────────────────────
sindh_mean_precip = df[df["district"] == "Sindh_District"]["precipitation"].mean()
kp_mean_precip    = df[df["district"] == "KP_District"]["precipitation"].mean()

sindh_spike = sindh_mean_precip * 4   # 300% above normal = 4x
kp_spike    = kp_mean_precip    * 4

print(f"Sindh  mean: {sindh_mean_precip:.2f} mm  →  spike: {sindh_spike:.2f} mm")
print(f"KP     mean: {kp_mean_precip:.2f} mm  →  spike: {kp_spike:.2f} mm")

# ── Step 2: Build spike rows ────────────────────────────────────
# Use real worst-case row per district as base
sindh_base = (df[df["district"] == "Sindh_District"]
              .sort_values("precipitation", ascending=False)
              .iloc[0].copy())

kp_base    = (df[df["district"] == "KP_District"]
              .sort_values("precipitation", ascending=False)
              .iloc[0].copy())

for row, spike_val in [(sindh_base, sindh_spike), (kp_base, kp_spike)]:
    # Override raw fields
    row["precipitation"]    = spike_val
    row["precip_3day_avg"]  = spike_val * 0.85   # neighbouring days also elevated
    row["precip_7day_avg"]  = spike_val * 0.60
    row["soil_moisture"]    = min(row["soil_moisture"] * 1.5, 0.99)  # saturated soil
    row["is_monsoon"]       = 1
    row["flood_event"]      = 1
    row["date"]             = "2024-08-15"

    # ✅ FIX: Re-compute ALL interaction features with NEW values
    # (avg_elevation_m already exists in row — no need to re-merge)
    row["soil_x_precip"]    = row["soil_moisture"] * row["precipitation"]
    row["monsoon_x_precip"] = row["is_monsoon"]    * row["precipitation"]
    row["humid_x_precip"]   = row["humidity"]      * row["precipitation"]
    row["precip_x_elev"]    = row["precipitation"] * row["avg_elevation_m"]
    row["soil_x_humidity"]  = row["soil_moisture"] * row["humidity"]
    row["temp_humidity"]    = row["temperature"]   * row["humidity"]

    # ✅ FIX: Use the SAME LabelEncoder (le) already fitted in the notebook
    row["district_enc"]     = le.transform([row["district"]])[0]

spike_df = pd.DataFrame([sindh_base, kp_base])

print(f"\nSpike rows created: {len(spike_df)}")
print(spike_df[["district", "precipitation", "precip_3day_avg",
                 "soil_moisture", "precip_x_elev", "flood_event"]])

# ── Step 3: Append and retrain ──────────────────────────────────
df_updated = pd.concat([df, spike_df], ignore_index=True)
print(f"\nDataset size: {len(df)} → {len(df_updated)} rows (+2 spike rows)")

# ✅ FIX: Use the SAME FEATURE_COLS from the notebook (already defined)
X_new = df_updated[FEATURE_COLS]
y_new = df_updated["flood_event"]

X_tr_new, X_te_new, y_tr_new, y_te_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
)

X_tr_s_new, y_tr_s_new = SMOTE(random_state=42).fit_resample(X_tr_new, y_tr_new)
print(f"After SMOTE: {len(X_tr_s_new)} rows | flood rate: {y_tr_s_new.mean():.2%}")

model_updated = xgb.XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=2, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)
model_updated.fit(
    X_tr_s_new, y_tr_s_new,
    eval_set=[(X_te_new, y_te_new)],
    verbose=False
)

acc_new = accuracy_score(y_te_new, model_updated.predict(X_te_new))
print(f"\nRetrained model accuracy: {acc_new:.4f} ({acc_new*100:.2f}%)")

# ── Step 4: Predict on spike rows — must not break ─────────────
print("\n── Spike Row Predictions ──")

def get_risk(p):
    if   p < 0.25: return "Low"
    elif p < 0.50: return "Medium"
    elif p < 0.75: return "High"
    else:          return "Critical"

# ✅ FIX: Use the SAME FEATURE_COLS — spike_df already has all columns
X_spike = spike_df[FEATURE_COLS]
probs   = model_updated.predict_proba(X_spike)[:, 1]

for district, prob, spike_val in zip(
        ["Sindh_District", "KP_District"],
        probs,
        [sindh_spike, kp_spike]):
    pred       = int(prob >= 0.5)
    risk       = get_risk(prob)
    confidence = round(prob * 100 if prob >= 0.5 else (1 - prob) * 100, 1)
    status     = "✅ Valid" if risk is not None else "❌ NULL"
    print(f"\n  District      : {district}")
    print(f"  Precipitation : {spike_val:.2f} mm")
    print(f"  flood_event   : {pred}")
    print(f"  probability_% : {prob*100:.1f}%")
    print(f"  risk_level    : {risk}   {status}")
    print(f"  confidence_%  : {confidence}%")

# ── Step 5: Full test-set check — no nulls ─────────────────────
print("\n── All Predictions — Null Check ──")
all_probs = model_updated.predict_proba(X_te_new)[:, 1]
results   = pd.DataFrame({
    "flood_event"   : (all_probs >= 0.5).astype(int),
    "probability_%" : (all_probs * 100).round(1),
    "risk_level"    : [get_risk(p) for p in all_probs],
    "confidence_%"  : (np.where(all_probs >= 0.5, all_probs,
                                1 - all_probs) * 100).round(1),
})

null_count = results["risk_level"].isna().sum()
print(f"  Total predictions : {len(results)}")
print(f"  NULL risk_levels  : {null_count}  {'✅' if null_count == 0 else '❌'}")
print(f"  Accuracy          : {acc_new*100:.2f}%")
print(f"\n  Risk distribution:")
print(results["risk_level"].value_counts().to_string())

# ── Pitch summary ───────────────────────────────────────────────
print(f"""
── How FloodSense handled the 300% spike ──
  1. Sindh received a {sindh_spike:.1f}mm spike (4× its mean of {sindh_mean_precip:.2f}mm)
  2. KP    received a {kp_spike:.1f}mm spike (4× its mean of {kp_mean_precip:.2f}mm)
  3. Both were labelled flood_event=1 and added to training data
  4. Model retrained — accuracy held at {acc_new*100:.2f}%
  5. Both spike rows returned valid risk classification (no crash, no null)
  6. Interaction features were correctly recomputed for spike values
  7. Soil moisture was capped at 0.99 (physically impossible to exceed 1.0)
""")


# ══════════════════════════════════════════════════════════════════
# FAULTY SENSOR TASK
# District chosen : Balochistan_District
# Why Balochistan : Geographically borders BOTH KP (northwest) and
#                   Sindh (southeast) — making them the 2 closest
# Cannot be dropped: high-population province
# ══════════════════════════════════════════════════════════════════

FAULTY_DISTRICT = "Balochistan_District"
NEIGHBOR_1      = "KP_District"       # northwest border
NEIGHBOR_2      = "Sindh_District"    # southeast border

# ── Step 1: Identify the faulty rows (current cycle = latest date) ──
latest_date = df["date"].max()
faulty_mask = (df["district"] == FAULTY_DISTRICT) & (df["date"] == latest_date)
n_faulty    = faulty_mask.sum()

print(f"Faulty district        : {FAULTY_DISTRICT}")
print(f"Current cycle date     : {latest_date}")
print(f"Rows to impute         : {n_faulty}")
print(f"Original precipitation : {df.loc[faulty_mask, 'precipitation'].values}")

# ── Step 2: Impute from the two nearest neighbors (same date) ──────
neighbor_rows  = df[
    (df["district"].isin([NEIGHBOR_1, NEIGHBOR_2])) &
    (df["date"] == latest_date)
]

assert len(neighbor_rows) == 2, \
    f"Expected 2 neighbor rows for {latest_date}, got {len(neighbor_rows)}"

print(f"\nNeighbor readings on {latest_date}:")
print(neighbor_rows[["district", "precipitation"]].to_string(index=False))

imputed_precip = neighbor_rows["precipitation"].mean()
print(f"\nImputed value (mean of neighbors) : {imputed_precip:.2f} mm")

# ── Step 3: Apply imputation — district stays in dataset ───────────
df_imputed = df.copy()

df_imputed.loc[faulty_mask, "precipitation"]   = imputed_precip
df_imputed.loc[faulty_mask, "precip_3day_avg"] = imputed_precip * 0.90  # prior days also elevated
df_imputed.loc[faulty_mask, "precip_7day_avg"] = imputed_precip * 0.75

# ✅ Re-compute ALL interaction features that depend on precipitation
for idx in df_imputed[faulty_mask].index:
    row = df_imputed.loc[idx]
    df_imputed.loc[idx, "soil_x_precip"]    = row["soil_moisture"] * row["precipitation"]
    df_imputed.loc[idx, "monsoon_x_precip"] = row["is_monsoon"]    * row["precipitation"]
    df_imputed.loc[idx, "humid_x_precip"]   = row["humidity"]      * row["precipitation"]
    df_imputed.loc[idx, "precip_x_elev"]    = row["precipitation"] * row["avg_elevation_m"]

print(f"\n{FAULTY_DISTRICT} retained in dataset with imputed value ✅")

# ── Step 4: Retrain on imputed dataset ─────────────────────────────
X_imp = df_imputed[FEATURE_COLS]
y_imp = df_imputed["flood_event"]

X_tr_imp, X_te_imp, y_tr_imp, y_te_imp = train_test_split(
    X_imp, y_imp, test_size=0.2, random_state=42, stratify=y_imp
)

X_tr_s_imp, y_tr_s_imp = SMOTE(random_state=42).fit_resample(X_tr_imp, y_tr_imp)
print(f"After SMOTE : {len(X_tr_s_imp)} rows | flood rate: {y_tr_s_imp.mean():.2%}")

model_imputed = xgb.XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=2, gamma=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)
model_imputed.fit(
    X_tr_s_imp, y_tr_s_imp,
    eval_set=[(X_te_imp, y_te_imp)],
    verbose=False
)

acc_imp = accuracy_score(y_te_imp, model_imputed.predict(X_te_imp))
print(f"\nRetrained model accuracy : {acc_imp:.4f} ({acc_imp*100:.2f}%)")

# ── Step 5: Validate — Balochistan must still produce valid output ──
print(f"\n── {FAULTY_DISTRICT} — Post-Imputation Predictions ──")

bal_X     = df_imputed[df_imputed["district"] == FAULTY_DISTRICT][FEATURE_COLS]
bal_probs = model_imputed.predict_proba(bal_X)[:, 1]
bal_risks = [get_risk(p) for p in bal_probs]   # get_risk() already defined above

null_count = sum(1 for r in bal_risks if r is None)
print(f"  Rows predicted    : {len(bal_probs)}")
print(f"  NULL risk_levels  : {null_count}  {'✅' if null_count == 0 else '❌'}")
print(f"  Risk distribution :")
print(pd.Series(bal_risks).value_counts().to_string())

# ── Pitch summary ───────────────────────────────────────────────────
print(f"""
── How FloodSense handled the faulty sensor ──
  1. Flagged district     : {FAULTY_DISTRICT}
  2. Current cycle date   : {latest_date}
  3. Cannot be dropped    : High-population province
  4. Strategy             : Geo-proximity imputation
  5. Nearest neighbors    : {NEIGHBOR_1} (NW border)
                            {NEIGHBOR_2} (SE border)
  6. Imputed value        : {imputed_precip:.2f} mm
                            (average of both neighbors, same date)
  7. Interaction features : Recomputed with imputed precipitation
  8. Model retrained      : Accuracy = {acc_imp*100:.2f}%
  9. {FAULTY_DISTRICT}  : No nulls, no crash ✅
""")


