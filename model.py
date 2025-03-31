
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Generate synthetic dataset
def generate_data(n=3000):
    np.random.seed(42)
    data = {
        "ambient_temperature": np.random.uniform(10, 40, n),
        "Voltage_measured": np.random.uniform(3.0, 4.2, n),
        "Current_measured": np.random.uniform(0.5, 5.0, n),
        "Rectified_impedance": np.random.uniform(0.05, 0.8, n),
        "Re": np.random.uniform(0.05, 0.8, n),
        "Capacity": np.random.uniform(0.8, 4.0, n),
        "Current_charge": np.random.uniform(0.5, 5.0, n),
        "Voltage_charge": np.random.uniform(3.0, 4.2, n),
        "Rct": np.random.uniform(0.1, 1.5, n),
        "Type": np.random.choice(["Lithium-ion", "Nickel-metal hydride", "Lead-acid"], n)
    }
    df = pd.DataFrame(data)
    df["Efficiency"] = df["Voltage_measured"] / df["Current_measured"] * df["Capacity"]
    return df

# Generate data
df = generate_data()
X = df.drop(columns=["Efficiency", "Type"])  # Features
y = df["Efficiency"]  # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance: MAE={mae:.4f}, R2={r2:.4f}")

# Save model
joblib.dump(model, "battery_model.pkl")
print("Model saved as battery_model.pkl")