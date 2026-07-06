import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from datetime import datetime
import os
import csv
from style import CSS

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FloodSense — PDMA Early Warning",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "en": {
        "title": "FloodSense",
        "subtitle": "PDMA District Flood Risk Assessment — Pakistan",
        "instructions": "Instructions:",
        "inst_1": "1. Select a district and date.",
        "inst_2": "2. Input rainfall, soil, and surface water conditions.",
        "inst_3": "3. Use the Advanced Sensor Readings for precise predictions.",
        "inst_4": "4. View immediate risk level and affected population.",
        "data_export": "📊 Data Export",
        "export_btn": "📥 Export Assessment Results",
        "no_assessments": "No assessments logged yet.",
        "dist_cond": "District Conditions",
        "loc_time": "📍 Location & Time",
        "district": "District",
        "choose_dist": "Choose a district...",
        "obs_date": "Date of Observation",
        "core_cond": "🌧️ Core Conditions",
        "today_rain": "Today's Rainfall (mm)",
        "soil_cond": "Soil Condition",
        "surface_water": "Surface Water Visible?",
        "adv_sensors": "⚙️ Advanced Sensor Readings (For Full Model Accuracy)",
        "past_3d": "Past 3 Days Total Rain (mm)",
        "temp": "Temperature (°C)",
        "wind": "Wind Speed (m/s)",
        "past_7d": "Past 7 Days Total Rain (mm)",
        "avg_temp_3d": "Avg Temp Past 3 Days (°C)",
        "pressure": "Pressure (Pa)",
        "past_3d_soil": "Past 3 Days Soil",
        "humidity": "Humidity (%)",
        "evap": "Evaporation (mm)",
        "assess_btn": "Assess Flood Risk",
        "sensor_note": "<strong>Sensor notice:</strong> The rainfall sensor in this district is flagged as faulty. The reading has been estimated using the average of neighbouring districts (Sindh and KP) for accuracy.",
        "ready_to_assess": "Ready to Assess",
        "ready_desc": "Fill in the district conditions on the left and submit to generate a flood risk assessment.",
        "risk_suffix": "Risk",
        "certainty": "Certainty",
        "est_pop": "Est. Population at Risk",
        "sys_rel": "System Reliability",
        "rec_action": "Recommended Action",
        "details": "Assessment details:",
        "monsoon": "Monsoon season",
        "non_monsoon": "Non-monsoon",
        "footer": "FloodSense — PDMA Early Warning System",
        "last_updated": "Last updated",
        "trained_on": "Trained on",
        "observations": "observations",
        "yes": "Yes",
        "no": "No",
        "dry": "Dry",
        "moist": "Moist",
        "saturated": "Saturated",
        "likelihood": "Flood Likelihood"
    },
    "ur": {
        "title": "فلڈ سینس",
        "subtitle": "پی ڈی ایم اے ڈسٹرکٹ سیلاب کے خطرے کی تشخیص — پاکستان",
        "instructions": "ہدایات:",
        "inst_1": "1. ضلع اور تاریخ منتخب کریں۔",
        "inst_2": "2. بارش، مٹی، اور سطح کے پانی کی صورتحال درج کریں۔",
        "inst_3": "3. درست پیش گوئی کے لیے ایڈوانس سینسر ریڈنگز کا استعمال کریں۔",
        "inst_4": "4. فوری خطرے کی سطح اور متاثرہ آبادی دیکھیں۔",
        "data_export": "📊 ڈیٹا ایکسپورٹ",
        "export_btn": "تشخیص کے نتائج ڈاؤن لوڈ کریں",
        "no_assessments": "ابھی تک کوئی تشخیص ریکارڈ نہیں ہوئی۔",
        "dist_cond": "ضلعی حالات",
        "loc_time": "📍 مقام اور وقت",
        "district": "ضلع",
        "choose_dist": "ضلع منتخب کریں...",
        "obs_date": "مشاہدے کی تاریخ",
        "core_cond": "🌧️ بنیادی حالات",
        "today_rain": "آج کی بارش (ملی میٹر)",
        "soil_cond": "مٹی کی حالت",
        "surface_water": "کیا سطح پر پانی نظر آ رہا ہے؟",
        "adv_sensors": "⚙️ ایڈوانس سینسر ریڈنگز (مکمل ماڈل کی درستگی کے لیے)",
        "past_3d": "گزشتہ 3 دنوں کی کل بارش (ملی میٹر)",
        "temp": "درجہ حرارت (سیلسیس)",
        "wind": "ہوا کی رفتار (میٹر/سیکنڈ)",
        "past_7d": "گزشتہ 7 دنوں کی کل بارش (ملی میٹر)",
        "avg_temp_3d": "گزشتہ 3 دنوں کا اوسط درجہ حرارت",
        "pressure": "ہوا کا دباؤ (پاسکل)",
        "past_3d_soil": "گزشتہ 3 دنوں کی مٹی",
        "humidity": "نمی (%)",
        "evap": "تبخیر (ملی میٹر)",
        "assess_btn": "سیلاب کے خطرے کی تشخیص کریں",
        "sensor_note": "<strong>سینسر نوٹس:</strong> اس ضلع میں بارش کا سینسر خراب ہے۔ درستگی کے لیے پڑوسی اضلاع (سندھ اور کے پی) کا اوسط استعمال کیا گیا ہے۔",
        "ready_to_assess": "تشخیص کے لیے تیار",
        "ready_desc": "بائیں جانب ضلعی حالات درج کریں اور سیلاب کے خطرے کی تشخیص کے لیے بٹن دبائیں۔",
        "risk_suffix": "خطرہ",
        "certainty": "یقین دہانی",
        "est_pop": "متاثرہ آبادی کا تخمینہ",
        "sys_rel": "سسٹم کی قابل اعتمادی",
        "rec_action": "تجویز کردہ عمل",
        "details": "تشخیص کی تفصیلات:",
        "monsoon": "مون سون کا موسم",
        "non_monsoon": "غیر مون سون",
        "footer": "فلڈ سینس — پی ڈی ایم اے ارلی وارننگ سسٹم",
        "last_updated": "آخری اپ ڈیٹ",
        "trained_on": "تربیت یافتہ",
        "observations": "مشاہدات",
        "yes": "جی ہاں",
        "no": "نہیں",
        "dry": "خشک",
        "moist": "نم",
        "saturated": "سیر شدہ",
        "likelihood": "سیلاب کا امکان"
    }
}

def t(key):
    return TRANSLATIONS[st.session_state.lang].get(key, key)

# --- INJECT THEME & LANG ---
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""
    <script>
        document.body.setAttribute('data-theme', '{st.session_state.theme}');
        document.body.setAttribute('data-lang', '{st.session_state.lang}');
        var targets = document.querySelectorAll('.main, .stApp, [data-testid="stAppViewContainer"]');
        targets.forEach(function(el) {{
            el.setAttribute('data-theme', '{st.session_state.theme}');
            el.setAttribute('data-lang', '{st.session_state.lang}');
        }});
    </script>
    <div style="display:none;" id="theme-injector"></div>
""", unsafe_allow_html=True)

# CSS is already loaded, but we need to wrap the app in a div to apply the data attributes correctly if JS is slow
# Actually we can just inject a small style tag to set the variables globally based on session state
THEME_OVERRIDE = f"""
<style>
    .stApp, .main {{
        --theme: {st.session_state.theme};
        --lang: {st.session_state.lang};
    }}
    .main[data-theme="light"], body[data-theme="light"], .stApp {{
        --bg-main: #f8faf9;
        --bg-card: #ffffff;
        --text-primary: #0f2418;
        --text-secondary: #2d4a38;
        --text-muted: #5f7a6a;
        --border-color: #e2ebe6;
        --header-border: #d4e4da;
        --stat-bg: linear-gradient(145deg, #ffffff, #f3f8f5);
        --alert-critical-bg: #fffbfb;
        --alert-high-bg: #fffaf5;
        --placeholder-bg: linear-gradient(145deg, #ffffff, #f5faf7);
        --input-bg: #ffffff;
        --input-border: #d0ddd5;
    }}
    .main[data-theme="dark"], body[data-theme="dark"] {{
        --bg-main: #0c1a11;
        --bg-card: #12251a;
        --text-primary: #f0f7f2;
        --text-secondary: #a7c5b1;
        --text-muted: #6e937a;
        --border-color: #21432f;
        --header-border: #21432f;
        --stat-bg: linear-gradient(145deg, #12251a, #0c1a11);
        --alert-critical-bg: #12251a;
        --alert-high-bg: #12251a;
        --placeholder-bg: linear-gradient(145deg, #12251a, #0c1a11);
    }}
</style>
"""
st.markdown(THEME_OVERRIDE, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model("floodsense_model.json")
    return m

@st.cache_data
def load_metadata():
    with open("floodsense_metadata.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_references():
    elev = pd.read_csv("district_elevation_reference.csv")
    df = pd.read_csv("floodsense_training_data.csv")
    df["water_area_pct_change"] = df["water_area_pct_change"].replace([np.inf, -np.inf], np.nan)
    medians = df.select_dtypes(include=[np.number]).median()
    return elev, medians, df

@st.cache_data
def load_impact_data():
    return pd.read_csv("ndma_flood_impact_2022.csv")

try:
    model = load_model()
    meta = load_metadata()
    elev_ref, medians, training_df = load_references()
    impact_df = load_impact_data()
except Exception as e:
    st.error(f"System error — please refresh. ({e})")
    st.stop()

# --- MAPPINGS ---
district_to_region = {
    "Sindh_District": "Sindh", "Balochistan_District": "Balochistan",
    "KP_District": "KP", "Nowshera": "KP", "Jacobabad": "Sindh",
    "Sialkot": "Punjab",
}
ROGUE_DISTRICT = "Balochistan_District"
NEIGHBOR_DISTRICTS = ["Sindh_District", "KP_District"]

# --- HELPERS ---
def impute_rainfall_for_rogue(rain, district):
    if district == ROGUE_DISTRICT:
        nbr = training_df[training_df["district"].isin(NEIGHBOR_DISTRICTS)].groupby("district")["precipitation"].median()
        return float(nbr.mean()) if len(nbr) >= 2 else float(medians.get("precipitation", rain))
    return rain

def get_population_at_risk(district, risk):
    region = district_to_region.get(district)
    if not region: return None
    row = impact_df[impact_df["region"] == region]
    if row.empty: return None
    total = int(row["affected_population"].values[0])
    scale = {"Low": 0.05, "Medium": 0.20, "High": 0.55, "Critical": 1.0}
    return int(total * scale.get(risk, 0.1))

def fmt_pop(n):
    if n is None: return "N/A"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}K"
    return str(n)

def validate(district, rainfall):
    if district is None: return False, "Please select a district to begin."
    if rainfall > 500: return False, "Insufficient data — manual assessment recommended. Rainfall exceeds reliable sensor range (>500 mm)."
    if rainfall < 0: return False, "Rainfall cannot be negative. Please verify the sensor reading."
    return True, ""

def log_assessment(district, obs_date, inp, prob, risk):
    log_file = "results.csv"
    file_exists = os.path.isfile(log_file)
    
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "District": district,
        "Observation Date": obs_date.strftime("%Y-%m-%d"),
        "Assessed Risk Level": risk,
        "Confidence (%)": round((prob if prob >= 0.5 else 1 - prob) * 100, 1),
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
        "Interaction: Soil x Precip": round(inp.get("soil_x_precip", 0), 2),
        "Interaction: Monsoon x Precip": round(inp.get("monsoon_x_precip", 0), 2),
        "Interaction: Humid x Precip": round(inp.get("humid_x_precip", 0), 2),
        "Interaction: Precip x Elev": round(inp.get("precip_x_elev", 0), 2),
        "Interaction: Soil x Humid": round(inp.get("soil_x_humidity", 0), 2),
        "Interaction: Temp x Humid": round(inp.get("temp_humidity", 0), 2)
    }
    
    try:
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
    except Exception as e:
        print(f"Failed to write log: {e}")

# ── LAYOUT ──────────────────────────────────────
alert_slot = st.empty()

# Header with language toggle
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col1:
    st.markdown('<div class="logo-mark">FS</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown(f'<div class="header-text"><h1>{t("title")}</h1><p>{t("subtitle")}</p></div>', unsafe_allow_html=True)
with header_col3:
    lang_toggle_col1, lang_toggle_col2 = st.columns(2)
    with lang_toggle_col1:
        if st.button("🇬🇧 EN", use_container_width=True, key="header_lang_en"):
            st.session_state.lang = "en"
            st.rerun()
    with lang_toggle_col2:
        if st.button("🇵🇰 اردو", use_container_width=True, key="header_lang_ur"):
            st.session_state.lang = "ur"
            st.rerun()

with st.sidebar:
    st.markdown('<div class="brand-text">⚡ FloodSense v2.2</div>', unsafe_allow_html=True)
    
    # Theme Toggle
    st.markdown('<div class="section-title" style="margin-top: 0;">🎨 Theme</div>', unsafe_allow_html=True)
    if st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"), help="Switch to dark theme"):
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    
    st.markdown("---")
    st.markdown(f"**{t('instructions')}**")
    st.markdown(f"""
    {t('inst_1')}
    {t('inst_2')}
    {t('inst_3')}
    {t('inst_4')}
    """)
    st.markdown("---")
    st.markdown(f'<div class="section-title" style="margin-top: 0;">{t("data_export")}</div>', unsafe_allow_html=True)
    if os.path.exists("results.csv"):
        with open("results.csv", "rb") as f:
            st.download_button(
                label=t("export_btn"),
                data=f,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info(t("no_assessments"))

col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown(f'<div class="card" data-lang="{st.session_state.lang}"><div class="card-title">{t("dist_cond")}</div>', unsafe_allow_html=True)
    with st.form("risk_form", border=False):
        st.markdown(f'<div class="section-title">{t("loc_time")}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: district = st.selectbox(t("district"), options=elev_ref['district'].unique(), index=None, placeholder=t("choose_dist"))
        with c2: obs_date = st.date_input(t("obs_date"), value=datetime.now())
        
        st.markdown(f'<div class="section-title">{t("core_cond")}</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: rainfall = st.number_input(t("today_rain"), min_value=0.0, max_value=2000.0, value=10.0, step=0.1)
        with c2: soil_cond = st.selectbox(t("soil_cond"), [t("dry"), t("moist"), t("saturated")], index=1)
        with c3: visible_water = st.selectbox(t("surface_water"), [t("no"), t("yes")], index=0)
        
        with st.expander(t("adv_sensors"), expanded=False):
            ac1, ac2, ac3 = st.columns(3)
            with ac1: 
                precip_3d_total = st.number_input(t("past_3d"), min_value=0.0, value=30.0, step=0.1)
                temperature = st.number_input(t("temp"), min_value=-10.0, max_value=55.0, value=30.0, step=0.5)
                wind_speed = st.number_input(t("wind"), min_value=0.0, value=5.0, step=0.1)
            with ac2: 
                precip_7d_total = st.number_input(t("past_7d"), min_value=0.0, value=70.0, step=0.1)
                temp_3d_avg = st.number_input(t("avg_temp_3d"), min_value=-10.0, max_value=55.0, value=30.0, step=0.5)
                pressure = st.number_input(t("pressure"), min_value=80000.0, value=101000.0, step=100.0)
            with ac3:
                soil_3d = st.selectbox(t("past_3d_soil"), [t("dry"), t("moist"), t("saturated")], index=1)
                humidity = st.number_input(t("humidity"), min_value=0.0, max_value=100.0, value=60.0, step=1.0)
                evaporation = st.number_input(t("evap"), min_value=-50.0, value=5.0, step=0.1)
                
        submitted = st.form_submit_button(t("assess_btn"))

    if district == ROGUE_DISTRICT:
        st.markdown(f'<div class="sensor-note">{t("sensor_note")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    if submitted:
        ok, err = validate(district, rainfall)
        if not ok:
            st.markdown(f'<div class="validation-msg">{err}</div>', unsafe_allow_html=True)
        else:
            precip = impute_rainfall_for_rogue(float(rainfall), district)
            precip_3day_avg = float(precip_3d_total) / 3.0
            precip_7day_avg = float(precip_7d_total) / 7.0
            
            soil_map = {t("dry"): 0.08, t("moist"): 0.12, t("saturated"): 0.25}
            soil_val = soil_map[soil_cond]
            soil_3d_val = soil_map[soil_3d]
            water_val = 0.0 if visible_water == t("no") else 0.5
            is_monsoon = 1 if obs_date.month in [7, 8, 9] else 0
            day_of_year = obs_date.timetuple().tm_yday
            month = obs_date.month
            dist_enc_name = district
            if district == "Nowshera": dist_enc_name = "KP_District"
            elif district == "Jacobabad": dist_enc_name = "Sindh_District"
            elif district == "Sialkot": dist_enc_name = "KP_District" # Similar elevation/plain profile
            
            dist_enc = meta["district_encoding"].get(dist_enc_name, 0)
            elev_row = elev_ref.loc[elev_ref['district'] == district, 'avg_elevation_m']
            avg_elev = float(elev_row.values[0]) if len(elev_row) > 0 else 200.0

            # --- MODEL ROBUSTNESS CLIPPING ---
            # Clip inputs to the ranges used in the retrained consistent model
            precip_clipped = min(precip, 50.0)
            soil_clipped = min(soil_val, 0.5)
            soil_3d_clipped = min(soil_3d_val, 0.5)
            water_clipped = min(water_val, 10.0)
            p3_clipped = min(precip_3day_avg, 50.0)
            p7_clipped = min(precip_7day_avg, 50.0)

            inp = {
                "evaporation": float(evaporation), 
                "precipitation": precip_clipped,
                "pressure": float(pressure), 
                "soil_moisture": soil_clipped,
                "temperature": float(temperature), 
                "wind_speed": float(wind_speed),
                "humidity": float(humidity), 
                "precip_3day_avg": p3_clipped,
                "precip_7day_avg": p7_clipped, 
                "temp_3day_avg": float(temp_3d_avg), 
                "soil_3day_avg": soil_3d_clipped,
                "day_of_year": day_of_year, 
                "month": month, 
                "is_monsoon": is_monsoon,
                "water_area_pct_change": water_clipped, 
                "avg_elevation_m": avg_elev, 
                "district_enc": dist_enc,
            }
            inp["soil_x_precip"] = inp["soil_moisture"] * inp["precipitation"]
            inp["monsoon_x_precip"] = inp["is_monsoon"] * inp["precipitation"]
            inp["humid_x_precip"] = inp["humidity"] * inp["precipitation"]
            inp["precip_x_elev"] = inp["precipitation"] * inp["avg_elevation_m"]
            inp["soil_x_humidity"] = inp["soil_moisture"] * inp["humidity"]
            inp["temp_humidity"] = inp["temperature"] * inp["humidity"]

            X = pd.DataFrame([inp])[meta["feature_cols"]]
            prob = model.predict_proba(X)[0, 1]

            levels = [
                (0.25, "Low",      "risk-low",      "#16a34a", "کم خطرہ",     "Continue normal operations. Monitor routine weather updates from the Meteorological Department."),
                (0.50, "Medium",   "risk-medium",   "#ca8a04", "درمیانہ خطرہ", "Alert local response teams. Clear drainage channels and pre-position relief supplies."),
                (0.75, "High",     "risk-high",     "#ea580c", "زیادہ خطرہ",   "Issue evacuation advisory for low-lying areas. Open emergency shelters and notify hospitals."),
                (9.99, "Critical", "risk-critical",  "#dc2626", "انتہائی خطرہ", "Initiate emergency evacuation immediately. Deploy rescue teams and activate all emergency protocols."),
            ]
            for thresh, risk, rc, color, urdu, action in levels:
                if prob < thresh: break

            # Log the successful assessment
            log_assessment(district, obs_date, inp, prob, risk)

            pop = fmt_pop(get_population_at_risk(district, risk))
            confidence = (prob if prob >= 0.5 else 1 - prob) * 100
            sys_acc = meta.get('test_accuracy', 0) * 100
            pct = prob * 100

            # SVG ring gauge math
            radius = 70
            cx, cy = 120, 120
            circumference = 2 * 3.14159 * radius
            filled = circumference * prob
            gap = circumference - filled

            import math
            tick_positions = []
            for val in [0, 25, 50, 75, 100]:
                angle = -90 + (val / 100) * 360
                rad = math.radians(angle)
                ix = cx + (radius - 6) * math.cos(rad)
                iy = cy + (radius - 6) * math.sin(rad)
                ox = cx + (radius + 6) * math.cos(rad)
                oy = cy + (radius + 6) * math.sin(rad)
                lx = cx + (radius + 17) * math.cos(rad)
                ly = cy + (radius + 17) * math.sin(rad)
                tick_positions.append((ix, iy, ox, oy, lx, ly, val))

            ticks_svg = ""
            for ix, iy, ox, oy, lx, ly, val in tick_positions:
                ticks_svg += f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" class="ring-ticks"/>'
                ticks_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" class="tick-label">{val}</text>'

            if risk in ["High", "Critical"]:
                ac = "critical" if risk == "Critical" else "high"
                alert_slot.markdown(f"""
                <div class="alert-strip {ac}">
                    <div class="alert-icon">⚠</div>
                    <div class="alert-body">
                        <h3>ACTIVE ALERT — {risk.upper()} {t('risk_suffix').upper()} IN {district.replace('_', ' ').upper()}</h3>
                        <p>{action}</p>
                    </div>
                </div>""", unsafe_allow_html=True)

            action_bg = {"Low": "#f0fdf4", "Medium": "#fefce8", "High": "#fff7ed", "Critical": "#fef2f2"}

            st.markdown(f"""
            <div class="risk-badge {rc}">
                <p class="risk-level">{risk} {t('risk_suffix')}</p>
                <p class="risk-urdu">{urdu}</p>
            </div>

            <div class="ring-gauge">
                <svg width="240" height="240" viewBox="0 0 240 240">
                    <circle cx="{cx}" cy="{cy}" r="{radius}" class="ring-track"/>
                    <circle cx="{cx}" cy="{cy}" r="{radius}" class="ring-fill"
                            stroke="{color}" stroke-dasharray="{filled:.1f} {gap:.1f}"/>
                    {ticks_svg}
                    <text x="{cx}" y="{cy - 6}" class="ring-center-val" fill="{color}">{pct:.0f}%</text>
                    <text x="{cx}" y="{cy + 14}" class="ring-center-lbl">{t('likelihood')}</text>
                </svg>
            </div>

            <div class="stat-row">
                <div class="stat-item">
                    <p class="stat-val">{confidence:.0f}%</p>
                    <p class="stat-lbl">{t('certainty')}</p>
                </div>
                <div class="stat-item">
                    <p class="stat-val">{pop}</p>
                    <p class="stat-lbl">{t('est_pop')}</p>
                </div>
                <div class="stat-item">
                    <p class="stat-val">{sys_acc:.0f}%</p>
                    <p class="stat-lbl">{t('sys_rel')}</p>
                </div>
            </div>

            <div class="action-banner" style="background:{action_bg[risk] if st.session_state.theme == 'light' else 'var(--bg-card)'}; border-color: {color if st.session_state.theme == 'dark' else 'transparent'}">
                <div class="action-dot" style="background:{color};"></div>
                <p class="action-text"><strong>{t('rec_action')}:</strong> {action}</p>
            </div>

            <div class="details-row">
                <strong>{t('details')}</strong> {district.replace('_', ' ')} &middot;
                {obs_date.strftime('%d %b %Y')} &middot;
                {t('today_rain')}: {rainfall:.1f} mm{' (estimated)' if district == ROGUE_DISTRICT else ''} &middot;
                {t('soil_cond')}: {soil_cond} &middot; {t('surface_water')}: {visible_water} &middot;
                {'Elevation'}: {avg_elev:.0f}m &middot;
                {t('monsoon') if is_monsoon else t('non_monsoon')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="placeholder-card">
            <h3>{t('ready_to_assess')}</h3>
            <p>{t('ready_desc')}</p>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-footer">
    {t('footer')} &middot; {t('sys_rel')}: {meta.get('test_accuracy',0)*100:.0f}% &middot;
    {t('trained_on')} {meta.get('n_train',0)+meta.get('n_test',0):,} {t('observations')} &middot;
    {t('last_updated')}: {datetime.now().strftime('%d %b %Y')}
</div>""", unsafe_allow_html=True)
