# Telco Customer Churn Prediction – Full Stack ML App

<p align="center">
  <img src="https://img.shields.io/badge/ML--End2End-green?style=for-the-badge" alt="End-to-End ML" />
  <img src="https://img.shields.io/badge/FastAPI--Backend-blue?style=for-the-badge" alt="FastAPI Backend" />
  <img src="https://img.shields.io/badge/Streamlit--UI-red?style=for-the-badge" alt="Streamlit UI" />
  <img src="https://img.shields.io/badge/Deployment--Live-success?style=for-the-badge" alt="Live Deployment" />
</p>

---

## Project Overview

📊 A complete end-to-end data science project to predict customer churn using machine learning and SHAP explainability.

![Streamlit Badge](https://img.shields.io/badge/Deployed-Streamlit-blue?logo=streamlit&logoColor=white)
🚀 **[Click here to use the live app!](https://telco-churn-predictor-tool.streamlit.app/)**



It goes beyond Jupyter notebooks and delivers:

- End-to-end ML pipeline (cleaning → modeling → tuning → explainability)
- **Interactive web app** using Streamlit
- **REST API service** using FastAPI
- Deployment-ready structure for both UI and API

🔧 Designed as a portfolio-ready project to **showcase full-stack ML engineering** in placement rounds and real-world interviews.

---

## Key Highlights

- ✅ **Real Telco dataset** from Kaggle
- ✅ Feature Engineering (tenure buckets, service counts, log transforms)
- ✅ Preprocessing: Scaling, Imputation, Outlier handling
- ✅ ML Models: Logistic Regression, LightGBM, RandomForest
- ✅ Model Tuning & Cross-validation
- ✅ SHAP Explainability (Waterfall, Summary, Force plots)
- ✅ Live predictions via **Streamlit App**
- ✅ **FastAPI** backend for production-ready REST inference
- ✅ Modular code, production structure, and best practices

---

## 🔧 Tech Stack

| Layer            | Tools Used                            |
|------------------|----------------------------------------|
| 📊 Data          | `Pandas`, `NumPy`, `PostgreSQL`        |
| 🤖 Modeling      | `scikit-learn`, `LightGBM`             |
| 🧠 Explainability| `SHAP`                                 |
| 🔙 Backend API   | `FastAPI`, `Uvicorn`                   |
| 🎨 Frontend UI   | `Streamlit`                            |
| 🚀 Deployment    | `GitHub`, `Streamlit Cloud`, `Render`  |

---

## 📂 Project Structure
telco-churn-predictor/
├── app.py # Streamlit UI
├── main_notebook.ipynb # Full ML pipeline & SHAP
├── api/
│ └── main.py # FastAPI backend
├── models/
│ ├── lgbm_model_tuned.pkl
│ ├── scaler.pkl
│ ├── scaled_columns.pkl
│ └── feature_columns.pkl
├── requirements.txt
└── README.md

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hitesh-bhatnagar/Telco-Customer-Churn-Predictor-Full-Stack-ML-App-.git
cd telco-churn-app
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv

.venv\Scripts\activate  # On Windows

source .venv/bin/activate # On Linux/macOS

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App (UI)

```bash
streamlit run app.py
```

### 5. Run the FastAPI API (optional)

```bash
uvicorn api.main:app --reload
```

## SHAP Explainability (Sample)

This project leverages **SHAP (SHapley Additive exPlanations)** to provide transparent and actionable insights into model predictions:

- **Feature Importance:** Identify which features most influence churn risk.
- **Per-Prediction Explanation:** Visualize individual predictions using waterfall plots.
- **Dependence Plots:** Explore how feature values impact model output.
- **Force Plots:** Interactive, intuitive explanations for each customer.

---

## 🌐 Deployment Options

**Streamlit Cloud (UI):**
- Instantly host `app.py` directly from GitHub.
- Free, fast, and ideal for demos or sharing with stakeholders.

**Render / Railway (API):**
- Deploy `api/main.py` using Docker or Gunicorn for scalable REST API services.
- Perfect for backend ML engineers and production-grade API deployment.

---

## 🙌 Acknowledgements

- **Dataset:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspired by real-world enterprise ML workflows.

---

## ⭐ Show Your Support

If you found this project valuable:

- 🌟 Star the repository
- 🍴 Fork and build your own solution
- 🐛 Report bugs or suggest improvements

## 👨‍💻 Author

**Hitesh Bhatnagar**  
B.Tech ECE • Data Science & ML Engineer  
[🔗 LinkedIn](https://www.linkedin.com/in/hitesh-bhatnagar-5a3b391ba) | [💻 GitHub](https://github.com/hitesh-bhatnagar)
