import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Trained Model
try:
    model = joblib.load("battery_model.pkl")
except Exception as e:
    st.error("Error loading model: " + str(e))
    st.stop()

# Background Image (EV Vehicle GIF)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: url("https://media.tenor.com/ItkYzjSCV1sAAAAC/electric-car-tesla.gif") no-repeat center fixed;
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar
#st.sidebar.image("your_image_path.jpg", use_container_width=True)  # Replace with actual image path
st.sidebar.header("OptiVolt - EV Battery Optimization")
st.sidebar.markdown(
    """
    **Real-time battery efficiency prediction**  
    Optimize battery performance with AI-driven insights.
    """
)

# Main Title & Styling
st.markdown("""
    <style>
        .big-title { font-size: 42px; font-weight: bold; text-align: center; color: #e74c3c; }
        .sub-title { font-size: 24px; text-align: center; color: #34495e; }
        .metric-box { 
            background-color: rgba(255, 255, 255, 0.8); 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            font-weight: bold; 
            margin-bottom: 15px;
        }
        .team-box {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            text-align: right;
            font-size: 14px;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">OptiVolt - EV Battery Efficiency Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enhancing battery performance with AI</p>', unsafe_allow_html=True)

# User Input Section
st.subheader("🔧 Enter Battery Parameters")

ambient_temperature = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0, step=0.5)
voltage_measured = st.slider("Voltage Measured (V)", 3.0, 4.2, 3.7, step=0.1)
current_measured = st.slider("Current Measured (A)", 0.5, 5.0, 2.5, step=0.1)
rectified_impedance = st.slider("Rectified Impedance (Ω)", 0.05, 0.8, 0.4, step=0.05)
Re = st.slider("Re (Ω)", 0.05, 0.8, 0.4, step=0.05)
capacity = st.slider("Capacity (Ah)", 0.8, 4.0, 2.0, step=0.1)
current_charge = st.slider("Current Charge (A)", 0.5, 5.0, 2.5, step=0.1)
voltage_charge = st.slider("Voltage Charge (V)", 3.0, 4.2, 3.7, step=0.1)
Rct = st.slider("Rct (Ω)", 0.1, 1.5, 0.8, step=0.05)

# Prepare the feature vector
input_features = np.array([[  
    ambient_temperature, voltage_measured, current_measured,  
    rectified_impedance, Re, capacity,  
    current_charge, voltage_charge, Rct  
]])

# Prediction Button
if st.button("🔍 Predict Efficiency"):
    # Make prediction
    prediction = model.predict(input_features)[0]
    st.markdown('<div class="metric-box">Predicted Battery Efficiency: {:.4f}</div>'.format(prediction), unsafe_allow_html=True)

    # Visualization Section
    st.markdown("---")
    st.subheader("📊 Data Visualizations")

    # 1. Line Chart: Efficiency vs. Voltage Measured
    voltage_range = np.linspace(3.0, 4.2, 100)
    efficiency = voltage_range / current_measured * capacity
    line_data = pd.DataFrame({"Voltage_measured": voltage_range, "Efficiency": efficiency})
    st.markdown("**Efficiency Trend vs Voltage Measured**")
    st.line_chart(line_data.set_index("Voltage_measured"))

    # 2. Scatter Plot: Efficiency vs Current Measured
    current_vals = np.linspace(0.5, 5.0, 100)
    efficiency_scatter = (voltage_measured / current_vals) * capacity
    scatter_df = pd.DataFrame({
        "Current_measured": current_vals,
        "Efficiency": efficiency_scatter
    })
    fig, ax = plt.subplots()
    ax.scatter(scatter_df["Current_measured"], scatter_df["Efficiency"], color="blue")
    ax.set_title("Efficiency vs Current Measured")
    ax.set_xlabel("Current Measured (A)")
    ax.set_ylabel("Efficiency")
    st.pyplot(fig, use_container_width=True)

    # 3. Bar Chart: Voltage & Current Affect on Capacity
    df_bar = pd.DataFrame({
        "Factor": ["Voltage Measured", "Current Measured"],
        "Value": [voltage_measured, current_measured]
    })
    st.bar_chart(df_bar.set_index("Factor"))

    # 4. Scatter Plot: Rct vs Ambient Temperature
    rct_values = np.linspace(0.1, 1.5, 100)
    ambient_temp_values = np.linspace(10.0, 40.0, 100)
    fig, ax = plt.subplots()
    ax.scatter(rct_values, ambient_temp_values, color="red")
    ax.set_title("Rct vs Ambient Temperature")
    ax.set_xlabel("Rct (Ω)")
    ax.set_ylabel("Ambient Temperature (°C)")
    st.pyplot(fig, use_container_width=True)

    # 5. Correlation Matrix (Fixed)
    correlation_data = pd.DataFrame({
        "Ambient Temp": np.random.uniform(10, 40, 100),
        "Voltage Measured": np.random.uniform(3.0, 4.2, 100),
        "Current Measured": np.random.uniform(0.5, 5.0, 100),
        "Rectified Impedance": np.random.uniform(0.05, 0.8, 100),
        "Re": np.random.uniform(0.05, 0.8, 100),
        "Capacity": np.random.uniform(0.8, 4.0, 100),
        "Current Charge": np.random.uniform(0.5, 5.0, 100),
        "Voltage Charge": np.random.uniform(3.0, 4.2, 100),
        "Rct": np.random.uniform(0.1, 1.5, 100),
        "Efficiency": np.random.uniform(0.75, 1.0, 100)
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig, use_container_width=True)

# Team Members Section
st.markdown('<div class="team-box">Group 10: Dhwanit Desai | Yash Gadhave | Jayesh Kandar</div>', unsafe_allow_html=True)
