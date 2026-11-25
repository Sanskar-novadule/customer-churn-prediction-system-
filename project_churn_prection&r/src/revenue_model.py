import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
from config import CLEANED_DATA_PATH


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(CLEANED_DATA_PATH)

print("Cleaned Data:")
print(df.head())

print("\nShape:")
print(df.shape)


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "Tenure",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic"
]

target = "TotalCharges"


X = df[features]
y = df[target]


print("\nSelected Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 3. CHECK FEATURES
# ============================================================

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nFeature Data Types:")
print(X.dtypes)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# 5. LINEAR REGRESSION MODEL
# ============================================================

linear_model = LinearRegression()

print("\nLinear Regression model created!")


# ============================================================
# 6. TRAIN LINEAR REGRESSION
# ============================================================

linear_model.fit(
    X_train,
    y_train
)

print("Linear Regression training completed!")


# ============================================================
# 7. LINEAR REGRESSION PREDICTION
# ============================================================

linear_pred = linear_model.predict(X_test)

print("\nActual Revenue:")
print(y_test.values)

print("\nLinear Regression Prediction:")
print(linear_pred)


# ============================================================
# 8. RANDOM FOREST REGRESSOR
# ============================================================

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\nRandom Forest Regressor model created!")


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

forest_model.fit(
    X_train,
    y_train
)

print("Random Forest Regressor training completed!")


# ============================================================
# 10. RANDOM FOREST PREDICTION
# ============================================================

forest_pred = forest_model.predict(X_test)

print("\nRandom Forest Prediction:")
print(forest_pred)

# ============================================================
# 11. LINEAR REGRESSION EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_pred
)

print("\n================================")
print("    LINEAR REGRESSION")
print("================================")

print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R2  :", linear_r2)

# ============================================================
# 12. RANDOM FOREST EVALUATION
# ============================================================

forest_mae = mean_absolute_error(
    y_test,
    forest_pred
)

forest_mse = mean_squared_error(
    y_test,
    forest_pred
)

forest_rmse = np.sqrt(forest_mse)

forest_r2 = r2_score(
    y_test,
    forest_pred
)

print("\n================================")
print("    RANDOM FOREST")
print("================================")

print("MAE :", forest_mae)
print("MSE :", forest_mse)
print("RMSE:", forest_rmse)
print("R2  :", forest_r2)

# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        forest_mae
    ],
    "MSE": [
        linear_mse,
        forest_mse
    ],
    "RMSE": [
        linear_rmse,
        forest_rmse
    ],
    "R2 Score": [
        linear_r2,
        forest_r2
    ]
})

print("\n================================")
print("    REVENUE MODEL COMPARISON")
print("================================")

print(results)