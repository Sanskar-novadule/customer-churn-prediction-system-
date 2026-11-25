import os

import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLEANED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_customer_churn.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(CLEANED_DATA_PATH)

except FileNotFoundError:

    st.error(
        f"Dataset not found:\n{CLEANED_DATA_PATH}"
    )

    st.stop()


# ============================================================
# REQUIRED FEATURES
# ============================================================

features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [
    "Churn",
    "TotalCharges"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Missing columns in dataset:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# CHURN MODEL
# ============================================================

X_churn = df[features]

y_churn = df["Churn"]


churn_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


churn_model.fit(
    X_churn,
    y_churn
)


# ============================================================
# REVENUE MODEL
# ============================================================

X_revenue = df[features]

y_revenue = df["TotalCharges"]


revenue_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


revenue_model.fit(
    X_revenue,
    y_revenue
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "Customer Churn Prediction & Revenue Forecasting"
)

st.write(
    "Enter customer details below to predict "
    "churn probability and estimated revenue."
)


st.divider()


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.subheader("👤 Customer Information")


col1, col2 = st.columns(2)


with col1:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


with col2:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0
    )


col3, col4 = st.columns(2)


with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with col4:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Customer",
    use_container_width=True
):

    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({

        "Tenure": [tenure],

        "MonthlyCharges": [monthly_charges],

        "Contract_One year": [
            1 if contract == "One year" else 0
        ],

        "Contract_Two year": [
            1 if contract == "Two year" else 0
        ],

        "InternetService_Fiber optic": [
            1 if internet_service == "Fiber optic" else 0
        ]

    })


    # ========================================================
    # CHURN PREDICTION
    # ========================================================

    churn_probability = churn_model.predict_proba(
        input_data
    )[0][1]


    churn_percentage = (
        churn_probability * 100
    )


    # ========================================================
    # CHURN PREDICTION CLASS
    # ========================================================

    churn_prediction = churn_model.predict(
        input_data
    )[0]


    if churn_prediction == 1:

        churn_result = "Yes"

    else:

        churn_result = "No"


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if churn_percentage < 30:

        risk_level = "Low Risk"

    elif churn_percentage < 60:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"


    # ========================================================
    # REVENUE PREDICTION
    # ========================================================

    predicted_revenue = revenue_model.predict(
        input_data
    )[0]


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("🎯 Prediction Result")


    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )


    with result_col2:

        st.metric(
            "Churn Prediction",
            churn_result
        )


    with result_col3:

        st.metric(
            "Predicted Revenue",
            f"${predicted_revenue:,.2f}"
        )


    with result_col4:

        st.metric(
            "Risk Level",
            risk_level
        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk_level == "High Risk":

        st.error(
            "⚠️ This customer has a HIGH probability "
            "of churning."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "⚠️ This customer has a MEDIUM probability "
            "of churning."
        )

    else:

        st.success(
            "✅ This customer has a LOW probability "
            "of churning."
        )


    # ========================================================
    # INPUT DATA DISPLAY
    # ========================================================

    st.subheader(
        "📋 Input Sent to ML Model"
    )


    st.dataframe(
        input_data,
        use_container_width=True
    )