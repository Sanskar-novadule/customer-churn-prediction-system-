import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
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
# 2. CREATE X AND Y
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nX:")
print(X.head())

print("\ny:")
print(y.head())

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

# ============================================================
# 4. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

print("\nLogistic Regression model created!")

# ============================================================
# 5. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("Model training completed!")
# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

print("\nActual:")
print(y_test.values)

print("\nPredicted:")
print(y_pred)

# ============================================================
# 7. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n================================")
print("     MODEL EVALUATION")
print("================================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# 9. DECISION TREE MODEL
# ============================================================

tree_model = DecisionTreeClassifier(
    random_state=42
)

print("\nDecision Tree model created!")

# Train
tree_model.fit(
    X_train,
    y_train
)

print("Decision Tree training completed!")

# Predict
tree_pred = tree_model.predict(X_test)

print("\nDecision Tree Actual:")
print(y_test.values)

print("\nDecision Tree Predicted:")
print(tree_pred)

tree_accuracy = accuracy_score(y_test, tree_pred)
tree_precision = precision_score(
    y_test,
    tree_pred,
    zero_division=0
)
tree_recall = recall_score(
    y_test,
    tree_pred,
    zero_division=0
)
tree_f1 = f1_score(
    y_test,
    tree_pred,
    zero_division=0
)

print("\n================================")
print("       DECISION TREE")
print("================================")

print("Accuracy :", tree_accuracy)
print("Precision:", tree_precision)
print("Recall   :", tree_recall)
print("F1 Score :", tree_f1)

tree_cm = confusion_matrix(
    y_test,
    tree_pred,
    labels=[0, 1]
)

print("\nDecision Tree Confusion Matrix:")
print(tree_cm)

# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nRandom Forest model created!")

# Train Random Forest

forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed!")

# Predict

forest_pred = forest_model.predict(X_test)

print("\nRandom Forest Actual:")
print(y_test.values)

print("\nRandom Forest Predicted:")
print(forest_pred)

# ============================================================
# 11. RANDOM FOREST EVALUATION
# ============================================================

forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

forest_precision = precision_score(
    y_test,
    forest_pred,
    zero_division=0
)

forest_recall = recall_score(
    y_test,
    forest_pred,
    zero_division=0
)

forest_f1 = f1_score(
    y_test,
    forest_pred,
    zero_division=0
)

print("\n================================")
print("       RANDOM FOREST")
print("================================")

print("Accuracy :", forest_accuracy)
print("Precision:", forest_precision)
print("Recall   :", forest_recall)
print("F1 Score :", forest_f1)

forest_cm = confusion_matrix(
    y_test,
    forest_pred,
    labels=[0, 1]
)

print("\nRandom Forest Confusion Matrix:")
print(forest_cm)

# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        tree_accuracy,
        forest_accuracy
    ],
    "Precision": [
        precision,
        tree_precision,
        forest_precision
    ],
    "Recall": [
        recall,
        tree_recall,
        forest_recall
    ],
    "F1 Score": [
        f1,
        tree_f1,
        forest_f1
    ]
})

print("\n================================")
print("       MODEL COMPARISON")
print("================================")

print(results)
# ============================================================
# 13. BEST CHURN MODEL
# ============================================================

best_model_index = results["F1 Score"].idxmax()

best_model_name = results.loc[
    best_model_index,
    "Model"
]

best_f1_score = results.loc[
    best_model_index,
    "F1 Score"
]

print("\n================================")
print("       BEST CHURN MODEL")
print("================================")

print("Model   :", best_model_name)
print("F1 Score:", best_f1_score)

# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

if best_model_name == "Logistic Regression":
    best_churn_model = model

elif best_model_name == "Decision Tree":
    best_churn_model = tree_model

else:
    best_churn_model = forest_model


print("\nSelected model for prediction:")
print(best_model_name)


# ============================================================
# 15. CHURN PROBABILITY
# ============================================================

churn_probability = best_churn_model.predict_proba(X)[:, 1]

print("\nChurn Probability:")
print(churn_probability)


# ============================================================
# 16. CHURN PREDICTION
# ============================================================

churn_prediction = best_churn_model.predict(X)

print("\nChurn Prediction:")
print(churn_prediction)

# ============================================================
# 17. RISK SEGMENTATION
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


print("\nRisk Levels:")
print(risk_levels)

# ============================================================
# 18. CREATE CHURN REPORT
# ============================================================

churn_report = pd.DataFrame()

if "CustomerID" in df.columns:
    churn_report["CustomerID"] = df["CustomerID"]

churn_report["Churn Probability"] = (
    churn_probability * 100
).round(2)

churn_report["Churn Prediction"] = churn_prediction

churn_report["Risk Level"] = risk_levels


print("\n==============================================")
print("           CUSTOMER CHURN REPORT")
print("==============================================")

print(churn_report)