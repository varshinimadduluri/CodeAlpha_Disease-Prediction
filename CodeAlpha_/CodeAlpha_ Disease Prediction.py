from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset (NO FILE REQUIRED)
data = load_breast_cancer()

X = data.data
y = data.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("\n=== Disease Prediction System ===")
print("Enter patient values:\n")

# Runtime input (ONLY FEW VALUES FOR SIMPLICITY)
# We will use first 5 features only
feature_names = data.feature_names[:5]

user_input = []

for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Add remaining features as average values (to avoid 30 inputs)
import numpy as np
avg_values = np.mean(X, axis=0)[5:]

full_input = np.concatenate([user_input, avg_values])

# Scale input
full_input = scaler.transform([full_input])

# Predict
prediction = model.predict(full_input)

print("\n=======================")

if prediction[0] == 1:
    print("✅ Result: Benign (No Disease Detected)")
else:
    print("⚠️ Result: Malignant (Disease Detected)")