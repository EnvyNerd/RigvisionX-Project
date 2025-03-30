import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.pkl")

model = load_model()

# Streamlit UI
st.title("Reservoir Oil Production Prediction")
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
    
    # Generate updated visualization based on user input
    perm_values = np.linspace(permeability * 0.5, permeability * 1.5, 50)
    pressure_values = np.linspace(pressure * 0.8, pressure * 1.2, 50)
    predictions = [model.predict([[porosity, p, pr, water_saturation, (p * A * (pr - bottomhole_pressure)) / (mu * L)]])[0] for p, pr in zip(perm_values, pressure_values)]
    
    fig, ax = plt.subplots()
    ax.plot(perm_values, predictions, label="Production vs. Permeability", color='b')
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
    
    # Interactive 3D Reservoir Visualization
    x = np.random.uniform(0, 100, 100)
    y = np.random.uniform(0, 100, 100)
    z = np.random.uniform(0, 50, 100)
    permeability_values = np.random.uniform(0.1, 1000, 100)
    
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=5,
            color=permeability_values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Permeability (mD)")
        )
    )])
    
    fig_3d.update_layout(
        title="3D Reservoir Permeability Distribution",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Depth"
        )
    )
    
    st.plotly_chart(fig_3d)

st.write("---")
st.write("Developed by Reservoir ML Team 🚀")
