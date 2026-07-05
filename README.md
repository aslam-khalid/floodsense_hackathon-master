<![CDATA[<div align="center">

# 🌊 FloodSense — PDMA Early Warning System

**AI-powered flood risk assessment for Pakistani districts**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**🔴 Live App →** [Click here to access FloodSense](https://YOUR-APP-NAME.streamlit.app)

</div>

---

## 🏆 About

FloodSense is an **early warning system** built during the **Neural Nova 48-Hour Data Drop Sprint** hackathon. It addresses Pakistan's critical need for timely flood warnings — a country where catastrophic flooding in 2022 submerged one-third of the nation, killed over 1,700 people, and displaced 32 million.

The system uses machine learning to predict flood risk at the district level, providing actionable alerts to non-technical PDMA officials with **zero setup** and **zero jargon**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **XGBoost ML Model** | Trained on 3,000+ daily sensor records with 80%+ accuracy |
| 🎯 **4-Tier Risk Classification** | Low · Medium · High · Critical — with color-coded badges |
| 🌍 **Bilingual Interface** | Full English and Urdu support simultaneously |
| 🌙 **Dark / Light Mode** | Elegant green-themed UI with smooth theme toggling |
| 📊 **Population Impact Estimates** | Real NDMA 2022 flood impact data for affected population projections |
| 🔧 **Sensor Fault Handling** | Automatic imputation for faulty rainfall sensors (Balochistan) |
| 📥 **CSV Export** | Download all assessment results for offline analysis |
| ⚡ **Lightweight** | Loads in under 10 seconds — no heavy dependencies or map tiles |

---

## 🧠 How It Works

```
User Input                    ML Pipeline                     Output
┌─────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
│ District     │    │ Feature Engineering      │    │ Risk Level       │
│ Rainfall     │───▶│ Interaction Terms        │───▶│ Confidence Score │
│ Soil / Water │    │ XGBoost Classification   │    │ Population Est.  │
│ Date         │    │ Probability Calibration  │    │ Action Advisory  │
└─────────────┘    └──────────────────────────┘    └──────────────────┘
```

**Risk Thresholds:**
- 🟢 **Low** (`< 25%`) — Continue normal operations
- 🟡 **Medium** (`25–50%`) — Alert local response teams
- 🟠 **High** (`50–75%`) — Issue evacuation advisory
- 🔴 **Critical** (`> 75%`) — Initiate emergency evacuation

---

## 📁 Project Structure

```
floodsense_hackathon-master/
├── app.py                          # Streamlit web application
├── style.py                        # CSS theme (green & white palette)
├── floodsense_model.json           # Trained XGBoost model
├── floodsense_metadata.json        # Model metadata & feature columns
├── floodsense_wrapper.py           # Wrapper for model inference
├── notebook_code.py                # Full ML training pipeline
├── floodsense_training_data.csv    # Training dataset (daily sensor records)
├── district_elevation_reference.csv# District elevation data
├── ndma_flood_impact_2022.csv      # NDMA 2022 regional flood impact data
├── requirements.txt                # Python dependencies
├── create_model_pkl.py             # Script to generate model.pkl
├── train_submission_model.py       # Submission model training script
├── test_website_model.py           # Website model testing script
└── data_dictionary.txt             # Column descriptions for datasets
```

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/aslam-khalid/floodsense_hackathon-master.git
cd floodsense_hackathon-master

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io) with custom CSS
- **ML Model:** [XGBoost](https://xgboost.readthedocs.io) Classifier
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Class Balancing:** SMOTE (imbalanced-learn)
- **Visualization:** Matplotlib, Seaborn

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **80%+** |
| Class Balancing | SMOTE oversampling |
| Features | 23 engineered features including interaction terms |
| Validation | Stratified train/test split |

---

## 🌐 Datasets

1. **`floodsense_training_data.csv`** — Daily sensor records for Pakistani districts (2022–2024)
2. **`ndma_flood_impact_2022.csv`** — NDMA regional flood impact totals
3. **`district_elevation_reference.csv`** — Terrain elevation data per district

---

## 👥 Team

Built by **Team Neural Nova** during the 48-Hour Data Drop Sprint hackathon.

> *"Pakistan does not have a flooding problem. It has a WARNING problem."*

---

<div align="center">

**⚡ FloodSense — Because every hour of warning saves lives.**

</div>
]]>
