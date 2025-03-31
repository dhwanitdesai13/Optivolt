
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic dataset
def generate_data(n=3000):
    np.random.seed(42)
    data = {
        "ambient_temperature": np.random.uniform(10, 40, n),
        "voltage_measured": np.random.uniform(3.0, 4.2, n),
        "current_measured": np.random.uniform(0.5, 5.0, n),
        "rectified_impedance": np.random.uniform(0.05, 0.8, n),
        "Re": np.random.uniform(0.05, 0.8, n),
        "capacity": np.random.uniform(0.8, 4.0, n),
        "current_charge": np.random.uniform(0.5, 5.0, n),
        "voltage_charge": np.random.uniform(3.0, 4.2, n),
        "Rct": np.random.uniform(0.1, 1.5, n),
    }
    df = pd.DataFrame(data)
    df["Efficiency"] = df["voltage_measured"] / df["current_measured"] * df["capacity"]
    return df

# Generate and save data
df = generate_data()
csv_path = "battery_data.csv"  # Path for saving CSV
df.to_csv(csv_path, index=False)

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Battery Features")
plt.show()

# Scatter plot: Efficiency vs. Voltage Measured
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["voltage_measured"], y=df["Efficiency"])
plt.xlabel("Voltage Measured (V)")
plt.ylabel("Efficiency")
plt.title("Efficiency vs. Voltage Measured")
plt.show()

# Scatter plot: Efficiency vs. Current Measured
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["current_measured"], y=df["Efficiency"])
plt.xlabel("Current Measured (A)")
plt.ylabel("Efficiency")
plt.title("Efficiency vs. Current Measured")
plt.show()

# Bar plot: Absolute correlation of features with Efficiency
plt.figure(figsize=(8, 5))
correlation_abs = correlation_matrix["Efficiency"].abs().sort_values(ascending=False)[1:]
sns.barplot(x=correlation_abs.values, y=correlation_abs.index, palette="Blues_r")
plt.xlabel("Absolute Correlation with Efficiency")
plt.title("Feature Importance based on Correlation")
plt.show()

# Return CSV path for download
csv_path