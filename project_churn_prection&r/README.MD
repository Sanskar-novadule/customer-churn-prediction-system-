# 📊 Customer Churn Prediction & Revenue Forecasting

A complete end-to-end Machine Learning project that predicts customer churn probability, segments customers by risk level, and forecasts revenue — with an interactive Streamlit dashboard.

---

## 🚀 Project Overview

This project uses real customer telecom data to:

- Predict whether a customer will **churn (leave)** or stay
- Classify customers into **Low / Medium / High Risk** segments
- Forecast **expected revenue** per customer
- Visualize insights through an **interactive dashboard**

---

## 🗂️ Project Structure

```
project_churn_prection&r/
│
├── data/
│   ├── customer_churn.csv               # Raw original dataset
│   ├── cleaned_customer_churn.csv       # Preprocessed & encoded data
│   ├── customer_prediction_report.csv   # Per-customer predictions
│   └── final_customer_report.csv        # Final combined report
│
├── src/
│   ├── config.py                        # All file paths & settings
│   ├── data_loading.py                  # Load & inspect raw data
│   ├── eda.py                           # Exploratory Data Analysis
│   ├── preprocessing.py                 # Data cleaning & encoding
│   ├── churn_model.py                   # Train & compare churn models
│   ├── revenue_model.py                 # Train revenue prediction model
│   ├── customer_prediction.py           # Predict for each customer
│   ├── evaluation.py                    # Model evaluation metrics
│   └── final_report.py                  # Generate final report
│
├── dashboard/
│   ├── app.py                           # Streamlit prediction app
│   └── analytics.py                     # Streamlit analytics dashboard
│
├── outputs/
│   ├── plots/                           # Saved charts & graphs
│   └── predictions/                     # Saved prediction outputs
│
├── venv/                                # Virtual environment (not committed)
├── requirements.txt                     # Python dependencies
└── README.MD                            # Project documentation
```

---

## 🧠 ML Models Used

| Task | Models Tried | Best Model |
|---|---|---|
| Churn Prediction | Logistic Regression, Decision Tree, Random Forest | Random Forest |
| Revenue Forecasting | Linear Regression, Random Forest Regressor | Random Forest |

**Features used for prediction:**
- `Tenure` — How long the customer has been with the company
- `MonthlyCharges` — Monthly billing amount
- `Contract_One year` — One year contract (encoded)
- `Contract_Two year` — Two year contract (encoded)
- `InternetService_Fiber optic` — Fiber optic internet (encoded)

---

## ⚙️ Setup & Installation

### Step 1 — Create Virtual Environment

```powershell
cd "e:\softwears 2026\Novadule\project_churn_prection&r"
python -m venv venv
```

### Step 2 — Activate Virtual Environment

```powershell
venv\Scripts\Activate.ps1
```

> If you get a script execution error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install Dependencies

```powershell
pip install streamlit pandas scikit-learn seaborn matplotlib numpy
```

---

## ▶️ How to Run

### Run ML Pipeline (in order)

```powershell
cd "e:\softwears 2026\Novadule\project_churn_prection&r\src"

python data_loading.py        # Step 1: Load & inspect data
python eda.py                 # Step 2: Exploratory analysis
python preprocessing.py       # Step 3: Clean & encode data
python churn_model.py         # Step 4: Train churn models
python revenue_model.py       # Step 5: Train revenue model
python customer_prediction.py # Step 6: Predict for all customers
python evaluation.py          # Step 7: Evaluate model
python final_report.py        # Step 8: Generate final report
```

### Run Streamlit Dashboard

```powershell
cd "e:\softwears 2026\Novadule\project_churn_prection&r"
streamlit run dashboard/app.py
```

### Run Analytics Dashboard

```powershell
streamlit run dashboard/analytics.py
```

Dashboard opens at: **http://localhost:8501**

---

## 📋 Dataset Features

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Tenure` | Months as a customer |
| `MonthlyCharges` | Monthly billing amount ($) |
| `TotalCharges` | Total amount charged ($) |
| `Contract` | Month-to-month / One year / Two year |
| `InternetService` | DSL / Fiber optic / No |
| `Gender` | Male / Female |
| `PaymentMethod` | Payment type |
| `Churn` | Yes / No (target variable) |

---

## 🎯 Risk Segmentation

| Churn Probability | Risk Level |
|---|---|
| < 30% | ✅ Low Risk |
| 30% – 60% | ⚠️ Medium Risk |
| > 60% | 🔴 High Risk |

---

## 📊 Dashboard Features

**Prediction App (`app.py`)**
- Enter customer details manually
- Get instant churn probability
- See predicted revenue
- View risk level with color-coded alerts

**Analytics Dashboard (`analytics.py`)**
- Dataset overview & statistics
- Churn distribution charts
- Revenue analysis plots
- Customer segmentation visuals

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Library:** Scikit-learn
- **Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `data/cleaned_customer_churn.csv` | Preprocessed data ready for ML |
| `data/customer_prediction_report.csv` | Churn predictions per customer |
| `data/final_customer_report.csv` | Full report with risk levels |
| `outputs/plots/` | All generated charts |
| `outputs/predictions/` | Saved prediction results |

---

## 👨‍💻 Author

**Novadule**
Customer Churn Prediction & Revenue Forecasting System
Built with Python & Streamlit
