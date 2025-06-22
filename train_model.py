import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the training data
df = pd.read_csv("Train_Data.csv")

# Encode categorical variables
label_encoders = {}
for column in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "insurance_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model trained and saved.")
