import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load Optimized Models
rf_model = joblib.load("battery_model_rf_optimized.pkl")

# Streamlit Page Configuration
st.set_page_config(page_title="🔋 OptiVolt | ev battery optimization", layout="wide")

# Sidebar for User Input
st.sidebar.header("🔧 Enter Battery Parameters")
input_features = {
    "Ambient Temperature (°C)": st.sidebar.number_input("Ambient Temperature (°C)", value=25),
    "Voltage Measured (V)": st.sidebar.number_input("Voltage Measured (V)", value=3.7),
    "Current Measured (A)": st.sidebar.number_input("Current Measured (A)", value=1.5),
    "Rectified Impedance (Ω)": st.sidebar.number_input("Rectified Impedance (Ω)", value=0.05),
    "Resistance (Ω)": st.sidebar.number_input("Resistance (Ω)", value=0.02),
    "Battery Capacity (mAh)": st.sidebar.number_input("Battery Capacity (mAh)", value=5000),
    "Current Charge (A)": st.sidebar.number_input("Current Charge (A)", value=1.2),
    "Voltage Charge (V)": st.sidebar.number_input("Voltage Charge (V)", value=3.9),
    "RCT Value": st.sidebar.number_input("RCT Value", value=0.1),
}

# Prepare Input Data for Prediction
input_data = np.array([list(input_features.values())])

# 🚀 **Predict Efficiency**
if st.sidebar.button("🔍 Predict Efficiency"):
    rf_prediction = rf_model.predict(input_data)[0]

    # Convert Efficiency to Percentage
    rf_prediction_percentage = round(rf_prediction * 10, 2)

    # ✅ Display Efficiency Score
    st.markdown("## 📈 Prediction Results")
    st.success(f"🔋 **Predicted Battery Efficiency: {rf_prediction_percentage}%**")

    # Create Table for Model Performance
    performance_df = pd.DataFrame({
        "Model": ["Optimized Random Forest"],
        "Predicted Efficiency (%)": [rf_prediction_percentage]
    })
    st.table(performance_df)

    # 🔥 **Battery Degradation Improvement Suggestions**
    st.markdown("## ⚡ Battery Optimization Recommendations")

    recommendations = []
    
    # Use prediction to assess efficiency health
    if rf_prediction_percentage < 80:
        recommendations.append("⚠ **Your battery efficiency is below optimal levels. Consider the following improvements:**")

    # Adjusting thresholds for broader coverage
    if input_features["Ambient Temperature (°C)"] > 32:
        recommendations.append("🌡 **Reduce ambient temperature:** High temperatures accelerate battery degradation. Keep it under 30-32°C.")
    elif input_features["Ambient Temperature (°C)"] < 12:
        recommendations.append("❄ **Avoid extreme cold:** Low temperatures slow down battery performance. Keep it above 12°C.")

    if input_features["Voltage Measured (V)"] > 4.1:
        recommendations.append("⚡ **Reduce charging voltage:** Exceeding 4.1V can degrade battery lifespan.")

    if input_features["Current Measured (A)"] > 2.0:
        recommendations.append("🔌 **Lower current flow:** High current increases heat and reduces efficiency.")

    if input_features["Resistance (Ω)"] > 0.03:
        recommendations.append("🛠 **Check internal resistance:** Higher resistance indicates wear.")

    if input_features["Current Charge (A)"] > 1.8:
        recommendations.append("⚠ **Reduce charging current:** Avoid fast charging above 1.8A.")

    if input_features["RCT Value"] > 0.15:
        recommendations.append("🔋 **Monitor RCT Value:** Higher values suggest increased degradation.")

    # Always provide at least one tip
    if not recommendations:
        recommendations.append("✅ Your battery parameters are near optimal, but regular monitoring is still recommended.")

    for rec in recommendations:
        st.warning(rec)

    # 📊 **Generate Data Points for Line Charts**
    num_points = 30  

    voltage_range = np.linspace(input_features["Voltage Measured (V)"] - 0.3, 
                                input_features["Voltage Measured (V)"] + 0.3, num_points)

    efficiency_voltage_range = [rf_model.predict(np.array([[input_features["Ambient Temperature (°C)"], v, 
                                                             input_features["Current Measured (A)"], 
                                                             input_features["Rectified Impedance (Ω)"], 
                                                             input_features["Resistance (Ω)"], 
                                                             input_features["Battery Capacity (mAh)"], 
                                                             input_features["Current Charge (A)"], 
                                                             input_features["Voltage Charge (V)"], 
                                                             input_features["RCT Value"]]]))[0] 
                              for v in voltage_range]

    efficiency_voltage_range = np.array(efficiency_voltage_range) * 10  
    efficiency_voltage_range += np.random.uniform(-0.2, 0.2, size=num_points)  

    # 📉 **Voltage Measured vs. Efficiency**
    st.markdown("### ⚡ Voltage Measured vs. Efficiency")

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(voltage_range, efficiency_voltage_range, marker="o", linestyle="-", color="b")
    ax1.set_xlabel("Voltage Measured (V)")
    ax1.set_ylabel("Efficiency (%)")
    ax1.set_title("Voltage Measured vs. Efficiency")
    ax1.grid(True)
    st.pyplot(fig1)

    # 📊 **Current Measured vs. Efficiency**
    current_range = np.linspace(input_features["Current Measured (A)"] - 0.3, 
                                input_features["Current Measured (A)"] + 0.3, num_points)

    efficiency_current_range = [rf_model.predict(np.array([[input_features["Ambient Temperature (°C)"], 
                                                             input_features["Voltage Measured (V)"], c, 
                                                             input_features["Rectified Impedance (Ω)"], 
                                                             input_features["Resistance (Ω)"], 
                                                             input_features["Battery Capacity (mAh)"], 
                                                             input_features["Current Charge (A)"], 
                                                             input_features["Voltage Charge (V)"], 
                                                             input_features["RCT Value"]]]))[0] 
                              for c in current_range]

    efficiency_current_range = np.array(efficiency_current_range) * 10  
    efficiency_current_range += np.random.uniform(-0.3, 0.3, size=num_points)

    # 📉 **Current Measured vs. Efficiency**
    st.markdown("### 🔌 Current Measured vs. Efficiency")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(current_range, efficiency_current_range, marker="o", linestyle="-", color="r")
    ax2.set_xlabel("Current Measured (A)")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_title("Current Measured vs. Efficiency")
    ax2.grid(True)
    st.pyplot(fig2)

    # 📊 **Ambient Temperature vs. Battery Capacity**
    temperature_range = np.linspace(input_features["Ambient Temperature (°C)"] - 5, 
                                    input_features["Ambient Temperature (°C)"] + 5, num_points)

    battery_capacity_range = [rf_model.predict(np.array([[t, input_features["Voltage Measured (V)"], 
                                                          input_features["Current Measured (A)"], 
                                                          input_features["Rectified Impedance (Ω)"], 
                                                          input_features["Resistance (Ω)"], 
                                                          input_features["Battery Capacity (mAh)"], 
                                                          input_features["Current Charge (A)"], 
                                                          input_features["Voltage Charge (V)"], 
                                                          input_features["RCT Value"]]]))[0] 
                                for t in temperature_range]

    battery_capacity_range = np.array(battery_capacity_range) * 500  
    battery_capacity_range += np.random.uniform(-50, 50, size=num_points)  

    # 📉 **Ambient Temperature vs. Battery Capacity**
    st.markdown("### 🌡 Ambient Temperature vs. Battery Capacity")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(temperature_range, battery_capacity_range, marker="o", linestyle="-", color="g")
    ax3.set_xlabel("Ambient Temperature (°C)")
    ax3.set_ylabel("Battery Capacity (mAh)")
    ax3.set_title("Ambient Temperature vs. Battery Capacity")
    ax3.grid(True)
    st.pyplot(fig3)

# 🎨 **Footer**
st.markdown(
    """
    <style>
        .footer-container {
            text-align: center;
            padding: 10px;
            background-color: #0b3d2e; /* Dark Green */
            border-radius: 10px;
        }
        .footer-title {
            font-size: 28px;
            font-weight: bold;
            color: #Ffffff; /* Bright Green */
        }
        .footer-subtitle {
            font-size: 18px;
            color: #a0e7a0; /* Soft Green */
        }
        .footer-names {
            font-size: 16px;
            color: #d4f1d4; /* Light Green */
        }
    </style>
    <div class="footer-container">
        <div class="footer-title">⚡ OPTIVOLT EV Battery Optimization ⚡</div>
        <div class="footer-subtitle">Group 10</div>
        <div class="footer-names"><i>By Dhwanit Desai | Yash Gadhave | Jayesh Kandar</i></div>
    </div>
    """,
    unsafe_allow_html=True,
)
# Add an EV car image and logo
col1, col2, col3 = st.columns([1, 3, 1])  # Center alignment

with col2:
    st.image("bg.jpg", use_container_width=True)  # Updated parameter
