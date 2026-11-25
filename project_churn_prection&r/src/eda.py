import pandas as pd

from config import DATA_PATH


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Original Data:")
print(df.head())

# ============================================================
# 2. CHURN BY CONTRACT
# ============================================================

churn_by_contract = (
    df.groupby("Contract")["Churn"]
    .value_counts()
)

print("\nChurn By Contract:")
print(churn_by_contract)

# ============================================================
# 3. TENURE VS CHURN
# ============================================================

df["Churn_Yes"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

correlation = df["Tenure"].corr(df["Churn_Yes"])

print("\nTENURE VS CHURN")
print("Correlation:", correlation)

if correlation < 0:
    print("Higher tenure is associated with lower churn.")

elif correlation > 0:
    print("Higher tenure is associated with higher churn.")

else:
    print("No linear relationship between tenure and churn.")

# ============================================================
# 4. MONTHLY CHARGES VS CHURN
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()