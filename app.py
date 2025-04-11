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

# Base disease parameters
st.sidebar.subheader("Base Disease Parameters")
beta = st.sidebar.slider("Infection Rate (β)", 0.1, 0.5, 0.3, step=0.01)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 0.2, 0.1, step=0.01)
travel_prob = st.sidebar.slider("Base Travel Probability", 0.01, 0.2, 0.05, step=0.01)

# Community-specific parameters
st.sidebar.subheader("Community Characteristics")
community_params = []

for i in range(num_communities):
    st.sidebar.markdown(f"### Community {i+1}")

    # Create a container for each community's parameters
    community_container = st.sidebar.container()

    # Initial infected
    initial_infected = community_container.number_input(
        f"Initial Infected",
        min_value=0,
        max_value=population,
        value=1 if i == 0 else 0,
        key=f"initial_{i}",
    )

    # Quarantine effectiveness
    quarantine_effectiveness = community_container.slider(
        "Quarantine Effectiveness",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key=f"quarantine_{i}",
    )

    # Social distancing
    social_distancing = community_container.slider(
        "Social Distancing",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key=f"distancing_{i}",
    )

    # Testing rate
    testing_rate = community_container.slider(
        "Testing Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key=f"testing_{i}",
    )

    # Vaccination rate
    vaccination_rate = community_container.slider(
        "Vaccination Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key=f"vaccination_{i}",
    )

    # Travel restrictions
    travel_restrictions = community_container.slider(
        "Travel Restrictions",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key=f"travel_{i}",
    )

    community_params.append(
        {
            "quarantine_effectiveness": quarantine_effectiveness,
            "social_distancing": social_distancing,
            "testing_rate": testing_rate,
            "vaccination_rate": vaccination_rate,
            "travel_restrictions": travel_restrictions,
        }
    )

# Time range
simulation_days = st.sidebar.slider("Simulation Days", 50, 200, 100)
t = np.linspace(0, simulation_days, 1000)

# Calculate initial conditions
S0 = [population - initial_infected for i in range(num_communities)]
R0 = [0] * num_communities
initial_conditions = (
    np.array(S0 + [1 if i == 0 else 0 for i in range(num_communities)] + R0)
    / population
)

# Solve ODE
solution = odeint(
    multi_community_sir,
    initial_conditions,
    t,
    args=(beta, gamma, num_communities, travel_prob, population, community_params),
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
        f"Base Travel prob: {travel_prob}"
    )
    plt.text(
        0.75,
        0.95,
        param_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    st.pyplot(fig2)

# Display statistics and community characteristics
st.subheader("Simulation Statistics and Community Characteristics")
for i in range(num_communities):
    peak_infected = max(I[:, i])
    peak_day = t[np.argmax(I[:, i])]

    st.write(f"**Community {i+1}:**")
    st.write("**Characteristics:**")
    st.write(
        f"- Quarantine Effectiveness: {community_params[i]['quarantine_effectiveness']:.2f}"
    )
    st.write(f"- Social Distancing: {community_params[i]['social_distancing']:.2f}")
    st.write(f"- Testing Rate: {community_params[i]['testing_rate']:.2f}")
    st.write(f"- Vaccination Rate: {community_params[i]['vaccination_rate']:.2f}")
    st.write(f"- Travel Restrictions: {community_params[i]['travel_restrictions']:.2f}")

    st.write("**Outcomes:**")
    st.write(f"- Peak infection: {peak_infected:.0f} individuals")
    st.write(f"- Day of peak infection: {peak_day:.1f}")
    st.write(f"- Final recovered: {R[-1, i]:.0f} individuals")
    st.write("---")
