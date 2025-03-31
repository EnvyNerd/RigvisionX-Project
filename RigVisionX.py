import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.pkl")

model = load_model()

# Streamlit UI
st.title("RigVisionX: Oil Production Prediction")
st.markdown("Enter reservoir properties to predict oil production.")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
porosity = st.sidebar.slider("Porosity (Φ)", 0.05, 0.3, 0.15)
permeability = st.sidebar.slider("Permeability (mD)", 0.1, 1000.0, 100.0)
pressure = st.sidebar.slider("Initial Pressure (psi)", 1000, 5000, 3000)
water_saturation = st.sidebar.slider("Water Saturation (S_w)", 0.1, 0.6, 0.3)

# Calculate Flow Rate using Darcy’s Law
A = 500  # Cross-sectional area (ft^2)
mu = 1   # Oil viscosity (cP)
L = 50   # Reservoir length (ft)
bottomhole_pressure = 2000
flow_rate = (permeability * A * (pressure - bottomhole_pressure)) / (mu * L)

# Predict
if st.button("Predict Oil Production"):
    input_data = np.array([[porosity, permeability, pressure, water_saturation, flow_rate]])
    prediction = model.predict(input_data)[0]
    
    # Industry benchmark comparison
    avg_production = 500  # Assumed industry average
    feedback = "above average" if prediction > avg_production else "below average"
    
    st.success(f"Predicted Oil Production: {prediction:.2f} barrels/day")
    st.write(f"This is **{feedback}** compared to the industry benchmark of {avg_production} barrels/day.")
    
    # Gauge meter visualization
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicted Oil Production (barrels/day)"},
        gauge={
            "axis": {"range": [0, 1000]},
            "steps": [
                {"range": [0, avg_production * 0.8], "color": "red"},
                {"range": [avg_production * 0.8, avg_production * 1.2], "color": "yellow"},
                {"range": [avg_production * 1.2, 1000], "color": "green"}
            ],
        }
    ))
    st.plotly_chart(fig_gauge)
    
    # Generate updated visualization based on user input
    perm_values = np.linspace(permeability * 0.5, permeability * 1.5, 50)
    pressure_values = np.linspace(pressure * 0.8, pressure * 1.2, 50)
    predictions = []
    std_dev = []
    
    for p, pr in zip(perm_values, pressure_values):
        pred = model.predict([[porosity, p, pr, water_saturation, (p * A * (pr - bottomhole_pressure)) / (mu * L)]])[0]
        predictions.append(pred)
        std_dev.append(0.1 * pred)  # Assumed 10% standard deviation
    
    predictions = np.array(predictions)
    std_dev = np.array(std_dev)
    
    fig, ax = plt.subplots()
    ax.plot(perm_values, predictions, label="Production vs. Permeability", color='b')
    ax.fill_between(perm_values, predictions - std_dev, predictions + std_dev, color='b', alpha=0.2, label="Confidence Interval")
    ax.set_xlabel("Permeability (mD)")
    ax.set_ylabel("Predicted Oil Production (barrels/day)")
    ax.set_title("Impact of Permeability and Pressure on Oil Production")
    ax.legend()
    st.pyplot(fig)
    
    # Feature importance visualization
    if hasattr(model, 'feature_importances_'):
        feature_names = ["Porosity", "Permeability", "Pressure", "Water Saturation", "Flow Rate"]
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(feature_names, importances, color='teal')
        ax.set_xlabel("Feature Importance")
        ax.set_title("Model Feature Contributions")
        st.pyplot(fig)
    
    # Advanced 3D Reservoir Visualization with Layers and Well Paths
    x = np.random.uniform(0, 100, 200)
    y = np.random.uniform(0, 100, 200)
    z = np.random.uniform(0, 50, 200)
    permeability_values = np.random.uniform(0.1, 1000, 200)
    
    layers = np.linspace(0, 50, num=6)
    fig_3d = go.Figure()
    
    for i in range(len(layers) - 1):
        layer_mask = (z >= layers[i]) & (z < layers[i+1])
        fig_3d.add_trace(go.Scatter3d(
            x=x[layer_mask], y=y[layer_mask], z=z[layer_mask], mode='markers',
            marker=dict(
                size=5,
                color=permeability_values[layer_mask],
                colorscale='Viridis',
                opacity=0.8
            ),
            name=f"Layer {i+1}"
        ))
    
    # Add Well Path Visualization in the center
    well_x = np.linspace(50, 50, 10)
    well_y = np.linspace(50, 50, 10)
    well_z = np.linspace(0, 50, 10)
    fig_3d.add_trace(go.Scatter3d(
        x=well_x, y=well_y, z=well_z, mode='lines',
        line=dict(color='red', width=5), name="Well Path"
    ))
    
    # Update layout with separate legends and colorbars
    fig_3d.update_layout(
        title="3D Reservoir Permeability Distribution",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Depth"
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig_3d)
    
    # Report analysis
    st.subheader("Analysis Report")
    report_text = f"""
    - **Predicted Oil Production:** {prediction:.2f} barrels/day
    - **Industry Benchmark:** {avg_production} barrels/day
    - **Performance:** {feedback.capitalize()}
    - **Recommendation:**
        - If production is below average, consider enhancing reservoir stimulation.
        - If production is above average, maintain current operational strategies.
    """
    st.markdown(report_text)

st.write("---")
st.write("Developed by Witschi.Mihan 🚀")
