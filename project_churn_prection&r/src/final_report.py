import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from config import CLEANED_DATA_PATH


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(CLEANED_DATA_PATH)

print("Data loaded successfully!")

print("\nShape:")
print(df.shape)


# ============================================================
# 2. CHURN MODEL
# ============================================================

churn_features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]

X_churn = df[churn_features]
y_churn = df["Churn"]


# Train/Test split

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn,
    y_churn,
    test_size=0.2,
    random_state=42,
    stratify=y_churn
)


# Create model

churn_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train

churn_model.fit(
    X_train_churn,
    y_train_churn
)

print("\nChurn model trained!")


# ============================================================
# 3. CHURN PROBABILITY
# ============================================================

churn_probability = churn_model.predict_proba(
    X_churn
)[:, 1]


# ============================================================
# 4. REVENUE MODEL
# ============================================================

revenue_features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]

X_revenue = df[revenue_features]
y_revenue = df["TotalCharges"]


# Train/Test split

X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(
    X_revenue,
    y_revenue,
    test_size=0.2,
    random_state=42
)


# Create model

revenue_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train

revenue_model.fit(
    X_train_revenue,
    y_train_revenue
)

print("Revenue model trained!")


# ============================================================
# 5. PREDICT REVENUE
# ============================================================

predicted_revenue = revenue_model.predict(
    X_revenue
)


# ============================================================
# 6. RISK SEGMENTATION
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
# 7. CREATE FINAL REPORT
# ============================================================

final_report = pd.DataFrame()


# Customer ID if available

if "CustomerID" in df.columns:

    final_report["CustomerID"] = df["CustomerID"]


# Churn probability

final_report["Churn Probability"] = (
    churn_probability * 100
).round(2)


# Revenue

final_report["Predicted Revenue"] = (
    predicted_revenue
).round(2)


# Risk

final_report["Risk Level"] = risk_levels


# ============================================================
# 8. DISPLAY FINAL REPORT
# ============================================================

print("\n")
print("====================================================")
print("        CUSTOMER CHURN & REVENUE REPORT")
print("====================================================")

print(final_report)


# ============================================================
# 9. RISK SUMMARY
# ============================================================

print("\n")
print("====================================================")
print("                  RISK SUMMARY")
print("====================================================")

risk_summary = final_report["Risk Level"].value_counts()

print(risk_summary)


# ============================================================
# 10. CHURN SUMMARY
# ============================================================

high_risk_count = (
    final_report["Risk Level"] == "High Risk"
).sum()

medium_risk_count = (
    final_report["Risk Level"] == "Medium Risk"
).sum()

low_risk_count = (
    final_report["Risk Level"] == "Low Risk"
).sum()


print("\nHigh Risk Customers  :", high_risk_count)
print("Medium Risk Customers:", medium_risk_count)
print("Low Risk Customers   :", low_risk_count)


# ============================================================
# 11. REVENUE SUMMARY
# ============================================================

total_predicted_revenue = (
    final_report["Predicted Revenue"].sum()
)

high_risk_revenue = final_report.loc[
    final_report["Risk Level"] == "High Risk",
    "Predicted Revenue"
].sum()


print("\n")
print("====================================================")
print("                REVENUE SUMMARY")
print("====================================================")

print(
    "Total Predicted Revenue:",
    round(total_predicted_revenue, 2)
)

print(
    "High Risk Customer Revenue:",
    round(high_risk_revenue, 2)
)


# ============================================================
# 12. SAVE FINAL REPORT
# ============================================================

output_path = "../data/final_customer_report.csv"

final_report.to_csv(
    output_path,
    index=False
)

print("\n")
print("====================================================")
print("Final report saved!")
print("File:", output_path)
print("====================================================")