# Telco Customer Churn Prediction - Full Stack ML App

<p align="center">
  <img src="https://img.shields.io/badge/ML-FullStack-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Deployed-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge" />
</p>

---

## 📌 Overview

This project is a complete end-to-end **Telco Customer Churn Prediction** system designed for real-world data science pipelines.

We go beyond just training a model — we simulate a full **data science deployment environment** including:

- Realistic data ingestion and cleaning (PostgreSQL, CSV)
- Advanced feature engineering
- Multiple baseline and tuned ML models
- SHAP interpretability
- API development with **FastAPI**
- **Streamlit frontend UI**
- Final product deployment

This project is ready for deployment and can be showcased in portfolios or interviews.

---

## 🏗️ Features

✅ Real-world telecom churn dataset  
✅ PostgreSQL + CSV ingestion  
✅ Full preprocessing & outlier treatment  
✅ Advanced Feature Engineering  
✅ Logistic Regression, LightGBM, RandomForest  
✅ SHAP Explainability for feature impact  
✅ Trained model exposed as FastAPI endpoint  
✅ Streamlit UI frontend  
✅ Local and Cloud-ready  
✅ GitHub + Streamlit Cloud + Render deployable

---

## 🚀 Tech Stack

 Layer               | Tool                          

| 📊 Data           | Pandas, NumPy, PostgreSQL     |
| 🧠 ML Model       | Scikit-learn, LightGBM        |
| 🧪 Explainability | SHAP                          |
| 🔧 Backend        | FastAPI, Uvicorn              |
| 🎨 Frontend       | Streamlit                     |
| ☁️ Deployment     | GitHub, Streamlit Cloud       |

---

## 📂 Folder Structure
telco-churn-app/
├── api/ # FastAPI backend (prediction)
│ └── main.py
├── models/ # Saved model, scaler, column info
│ ├── lgbm_model_tuned.pkl
│ ├── scaler.pkl
│ └── feature_columns.pkl
├── app.py # Streamlit app (frontend)
└── README.md # Project documentation

---

## 🧪 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hitesh-bhatnagar/Telco-Customer-Churn-Predictor-Full-Stack-ML-App-
cd telco-churn-app
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/Mac
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend (in one terminal)

```bash
uvicorn api.main:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view the live Swagger API.

### 5. Start the Streamlit frontend (in another terminal)

```bash
streamlit run app.py
```

⭐ Show Your Support
If you found this project helpful:

🌟 Star this repo
🍴 Fork it
🐛 Submit issues
📢 Share with friends
