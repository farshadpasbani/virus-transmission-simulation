import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from main import multi_community_sir

st.set_page_config(page_title="Virus Transmission Simulation", layout="wide")

st.title("Multi-Community Virus Transmission Simulation")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Number of communities
num_communities = st.sidebar.slider("Number of Communities", 2, 5, 3)

# Population parameters
population = st.sidebar.slider("Population per Community", 100, 10000, 1000, step=100)

# Disease parameters
beta = st.sidebar.slider("Infection Rate (β)", 0.1, 0.5, 0.3, step=0.01)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 0.2, 0.1, step=0.01)
travel_prob = st.sidebar.slider("Travel Probability", 0.01, 0.2, 0.05, step=0.01)

# Initial conditions
st.sidebar.subheader("Initial Conditions")
initial_infected = []
for i in range(num_communities):
    initial_infected.append(
        st.sidebar.number_input(
            f"Initial Infected in Community {i+1}",
            min_value=0,
            max_value=population,
            value=1 if i == 0 else 0,
        )
    )

# Time range
simulation_days = st.sidebar.slider("Simulation Days", 50, 200, 100)
t = np.linspace(0, simulation_days, 1000)

# Calculate initial conditions
S0 = [population - initial_infected[i] for i in range(num_communities)]
R0 = [0] * num_communities
initial_conditions = np.array(S0 + initial_infected + R0) / population

# Solve ODE
solution = odeint(
    multi_community_sir,
    initial_conditions,
    t,
    args=(beta, gamma, num_communities, travel_prob, population),
)

# Extract results
S = solution[:, :num_communities] * population
I = solution[:, num_communities : 2 * num_communities] * population
R = solution[:, 2 * num_communities :] * population

# Create two columns for plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Individual Communities")
    fig1, axes = plt.subplots(num_communities, 1, figsize=(10, 4 * num_communities))
    if num_communities == 1:
        axes = [axes]

    for i in range(num_communities):
        axes[i].plot(t, S[:, i], "b", label="Susceptible")
        axes[i].plot(t, I[:, i], "r", label="Infected")
        axes[i].plot(t, R[:, i], "g", label="Recovered")
        axes[i].set_title(f"Community {i+1}")
        axes[i].set_ylabel("Number of Individuals")
        if i == num_communities - 1:
            axes[i].set_xlabel("Time (days)")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("Total Population")
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(t, np.sum(S, axis=1), "b", label="Total Susceptible")
    plt.plot(t, np.sum(I, axis=1), "r", label="Total Infected")
    plt.plot(t, np.sum(R, axis=1), "g", label="Total Recovered")
    plt.title("Total Population Across All Communities")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Individuals")
    plt.legend()
    plt.grid(True)

    # Add parameters text
    param_text = (
        f"Communities: {num_communities}\n"
        f"Pop per community: {population}\n"
        f"β: {beta}\n"
        f"γ: {gamma}\n"
        f"Travel prob: {travel_prob}"
    )
    plt.text(
        0.75,
        0.95,
        param_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    st.pyplot(fig2)

# Display statistics
st.subheader("Simulation Statistics")
for i in range(num_communities):
    peak_infected = max(I[:, i])
    peak_day = t[np.argmax(I[:, i])]
    st.write(f"**Community {i+1}:**")
    st.write(f"- Peak infection: {peak_infected:.0f} individuals")
    st.write(f"- Day of peak infection: {peak_day:.1f}")
    st.write(f"- Final recovered: {R[-1, i]:.0f} individuals")
