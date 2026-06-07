import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================
# SAMPLE CREDIT DATASET
# =====================================

data = {
    "Income": [50000,40000,60000,30000,70000,35000,80000,45000,28000,90000,
               55000,32000,75000,38000,85000,42000,26000,95000,52000,31000],

    "Debt": [10000,15000,5000,20000,7000,18000,4000,12000,22000,3000,
             8000,19000,5000,16000,3500,14000,24000,2000,9000,21000],

    "Payment_History": [1,0,1,0,1,0,1,1,0,1,
                        1,0,1,0,1,1,0,1,1,0],

    "Credit_Score": [750,650,800,600,820,620,850,700,580,870,
                     760,610,830,640,860,690,560,890,740,590],

    # 0 = Good Credit, 1 = Bad Credit
    "Risk": [0,1,0,1,0,1,0,0,1,0,
             0,1,0,1,0,0,1,0,0,1]
}

df = pd.DataFrame(data)

# =====================================
# FEATURE ENGINEERING
# =====================================

df["Debt_to_Income"] = df["Debt"] / df["Income"]

# =====================================
# FEATURES AND TARGET
# =====================================

X = df.drop("Risk", axis=1)
y = df["Risk"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# =====================================
# MODEL EVALUATION
# =====================================

for name, model in models.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy  : {accuracy:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")
    print(f"ROC-AUC   : {roc_auc:.2f}")

# =====================================
# USER PREDICTION
# =====================================

print("\n--- Credit Risk Prediction ---")

income = float(input("Enter Income: "))
debt = float(input("Enter Debt: "))
payment_history = int(input("Payment History (1=Good, 0=Bad): "))
credit_score = int(input("Credit Score: "))

debt_to_income = debt / income

new_customer = pd.DataFrame({
    "Income": [income],
    "Debt": [debt],
    "Payment_History": [payment_history],
    "Credit_Score": [credit_score],
    "Debt_to_Income": [debt_to_income]
})

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

final_model.fit(X, y)

prediction = final_model.predict(new_customer)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("Customer is Creditworthy (Low Risk)")
else:
    print("Customer is High Risk")