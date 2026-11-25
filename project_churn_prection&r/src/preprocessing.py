import pandas as pd
from config import DATA_PATH, CLEANED_DATA_PATH
from config import DATA_PATH, CLEANED_DATA_PATH


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Original Data:")
print(df.head())


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. CHECK DUPLICATE ROWS
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. CHECK DATA TYPES
# ============================================================

print("\nData Types:")
print(df.dtypes)

df = df.drop(columns=["CustomerID"])

print("\nAfter removing CustomerID:")
print(df.head())


# ============================================================
# 6. CATEGORICAL COLUMNS
# ============================================================

categorical_cols = [
    "Gender",
    "Contract",
    "InternetService",
    "PaymentMethod"
]

print("\nCategorical Columns:")
print(categorical_cols)

# ============================================================
# 7. CONVERT CHURN INTO 0 / 1
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn after encoding:")
print(df["Churn"])

# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

print("\nEncoded Data:")
print(df_encoded.head())

print("\nEncoded Columns:")
print(df_encoded.columns.tolist())

# ============================================================
# 9. SAVE CLEANED DATA
# ============================================================

df_encoded.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

print("\nCleaned data saved successfully!")
print("Saved to:", CLEANED_DATA_PATH)