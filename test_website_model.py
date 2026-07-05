import json
import pandas as pd
import xgboost as xgb

import importlib.util
spec = importlib.util.spec_from_file_location('run_prediction', 'run_prediction (1).py')
run_pred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_pred)
SCENARIOS = run_pred.SCENARIOS

# 2. Load the website's original 23-feature model
model = xgb.XGBClassifier()
model.load_model("floodsense_model.json")

with open("floodsense_metadata.json", "r") as f:
    meta = json.load(f)

# District elevation reference from app.py
elev_ref = {
    "Sindh_District": 34.0,
    "Balochistan_District": 1500.0,
    "KP_District": 1200.0,
    "Nowshera": 280.0,
    "Jacobabad": 55.0
}

def prob_to_risk(prob):
    if prob < 0.25: return "Low"
    elif prob < 0.50: return "Medium"
    elif prob < 0.75: return "High"
    else: return "Critical"

print("="*60)
print("  FLOODSENSE — WEBSITE MODEL (23-FEATURE) EVALUATION")
print("="*60)
print(f"  {'District':<25} {'Risk':<10} {'Confidence (App Logic)':>22}")
print("  " + "-"*65)

results = []
for scenario in SCENARIOS:
    district = scenario["district"]
    
    # App.py imputation logic for missing districts
    dist_enc_name = district
    if district == "Nowshera": dist_enc_name = "KP_District"
    if district == "Jacobabad": dist_enc_name = "Sindh_District"
    
    # 3. Construct the input dictionary exactly like app.py does
    inp = {
        "evaporation": scenario["evaporation"],
        "precipitation": scenario["precipitation"],
        "pressure": scenario["pressure"],
        "soil_moisture": scenario["soil_moisture"],
        "temperature": scenario["temperature"],
        "wind_speed": scenario["wind_speed"],
        "humidity": scenario["humidity"],
        "precip_3day_avg": scenario["precip_3day_avg"],
        "precip_7day_avg": scenario["precip_7day_avg"],
        "temp_3day_avg": scenario.get("temp_3day_avg", scenario["temperature"]), # App fallback
        "soil_3day_avg": scenario["soil_3day_avg"],
        "day_of_year": scenario["day_of_year"],
        "month": scenario["month"],
        "is_monsoon": scenario["is_monsoon"],
        "water_area_pct_change": scenario["water_area_pct_change"],
        "avg_elevation_m": elev_ref.get(district, 200.0),
        "district_enc": meta["district_encoding"].get(dist_enc_name, 0)
    }
    
    # Add interaction features exactly like app.py
    inp["soil_x_precip"] = inp["soil_moisture"] * inp["precipitation"]
    inp["monsoon_x_precip"] = inp["is_monsoon"] * inp["precipitation"]
    inp["humid_x_precip"] = inp["humidity"] * inp["precipitation"]
    inp["precip_x_elev"] = inp["precipitation"] * inp["avg_elevation_m"]
    inp["soil_x_humidity"] = inp["soil_moisture"] * inp["humidity"]
    inp["temp_humidity"] = inp["temperature"] * inp["humidity"]
    
    # Format exactly as expected by the model
    X = pd.DataFrame([inp])[meta["feature_cols"]]
    
    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    risk = prob_to_risk(prob)
    
    # App.py uses this specific confidence calculation
    confidence = (prob if prob >= 0.5 else 1 - prob) * 100
    
    print(f"  {district:<25} {risk:<10} {confidence:>21.1f}%")
    
    # Format nicely for the CSV export
    formatted_inp = {
        "District": district,
        "Assessed Risk Level": risk,
        "Confidence (%)": round(confidence, 1),
        "Flood Probability": round(prob, 4),
        "Precipitation (mm)": inp["precipitation"],
        "Temperature (°C)": inp["temperature"],
        "Humidity (%)": inp["humidity"],
        "Soil Moisture": inp["soil_moisture"],
        "Pressure (Pa)": inp["pressure"],
        "Wind Speed (m/s)": inp["wind_speed"],
        "Evaporation (mm)": inp["evaporation"],
        "Precipitation 3-Day Avg (mm)": inp["precip_3day_avg"],
        "Precipitation 7-Day Avg (mm)": inp["precip_7day_avg"],
        "Temperature 3-Day Avg (°C)": inp["temp_3day_avg"],
        "Soil Moisture 3-Day Avg": inp["soil_3day_avg"],
        "Water Area Change (%)": inp["water_area_pct_change"],
        "Avg Elevation (m)": inp["avg_elevation_m"],
        "Is Monsoon Season": "Yes" if inp["is_monsoon"] else "No",
        "Month": inp["month"],
        "Day of Year": inp["day_of_year"],
        "Interaction: Soil x Precip": round(inp["soil_x_precip"], 2),
        "Interaction: Monsoon x Precip": round(inp["monsoon_x_precip"], 2),
        "Interaction: Humid x Precip": round(inp["humid_x_precip"], 2),
        "Interaction: Precip x Elev": round(inp["precip_x_elev"], 2),
        "Interaction: Soil x Humid": round(inp["soil_x_humidity"], 2),
        "Interaction: Temp x Humid": round(inp["temp_humidity"], 2)
    }
    results.append(formatted_inp)

# Output test scenarios to results.csv
df = pd.DataFrame(results)
df.to_csv("test_scenarios_results.csv", index=False)
print("\n  Full feature assessments saved to test_scenarios_results.csv")
print("="*60)
