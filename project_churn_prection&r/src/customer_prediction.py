import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from config import CLEANED_DATA_PATH


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(CLEANED_DATA_PATH)

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# 2. CHURN FEATURES
# ============================================================

churn_features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]

churn_target = "Churn"


X_churn = df[churn_features]
y_churn = df[churn_target]


# ============================================================
# 3. TRAIN / TEST SPLIT FOR CHURN
# ============================================================

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn,
    y_churn,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. CHURN MODEL
# ============================================================

churn_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

churn_model.fit(
    X_train_churn,
    y_train_churn
)

print("\nChurn model trained successfully!")


# ============================================================
# 5. CHURN PROBABILITY
# ============================================================

churn_probability = churn_model.predict_proba(
    X_churn
)[:, 1]


# ============================================================
# 6. REVENUE FEATURES
# ============================================================

revenue_features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]

revenue_target = "TotalCharges"


X_revenue = df[revenue_features]
y_revenue = df[revenue_target]


# ============================================================
# 7. TRAIN / TEST SPLIT FOR REVENUE
# ============================================================

X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(
    X_revenue,
    y_revenue,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 8. REVENUE MODEL
# ============================================================

revenue_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

revenue_model.fit(
    X_train_revenue,
    y_train_revenue
)

print("Revenue model trained successfully!")


# ============================================================
# 9. REVENUE PREDICTION
# ============================================================

predicted_revenue = revenue_model.predict(
    X_revenue
)


# ============================================================
# 10. RISK SEGMENTATION
# ============================================================

def get_risk_level(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


risk_levels = [
    get_risk_level(probability)
    for probability in churn_probability
]


# ============================================================
# 11. CREATE FINAL REPORT
# ============================================================

result = pd.DataFrame()

# Customer ID
if "CustomerID" in df.columns:
    result["CustomerID"] = df["CustomerID"]

result["Churn Probability"] = (
    churn_probability * 100
).round(2)

result["Predicted Revenue"] = (
    predicted_revenue
).round(2)

result["Risk Level"] = risk_levels


# ============================================================
# 12. DISPLAY FINAL REPORT
# ============================================================

print("\n==============================================")
print("       CUSTOMER CHURN & REVENUE REPORT")
print("==============================================")

print(result)


# ============================================================
# 13. RISK SUMMARY
# ============================================================

print("\n==============================================")
print("                 RISK SUMMARY")
print("==============================================")

print(
    result["Risk Level"].value_counts()
)


# ============================================================
# 14. SAVE FINAL REPORT
# ============================================================

output_path = "../data/customer_prediction_report.csv"

result.to_csv(
    output_path,
    index=False
)

print("\n==============================================")
print("Final report saved successfully!")
print("File:", output_path)
print("==============================================")