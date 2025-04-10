import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def multi_community_sir(y, t, beta, gamma, num_communities, travel_prob, population):
    # Unpack the state variables
    S = y[:num_communities]
    I = y[num_communities : 2 * num_communities]
    R = y[2 * num_communities :]

    dSdt = np.zeros(num_communities)
    dIdt = np.zeros(num_communities)
    dRdt = np.zeros(num_communities)

    # Calculate infections within and between communities
    for i in range(num_communities):
        # Within community infection
        local_infection = beta * S[i] * I[i]

        # Between community infections (travel)
        travel_infection = 0
        for j in range(num_communities):
            if i != j:
                # Probability of travel and catching infection from other community
                travel_infection += travel_prob * beta * S[i] * I[j]

        # Total change rates
        dSdt[i] = -local_infection - travel_infection
        dIdt[i] = local_infection + travel_infection - gamma * I[i]
        dRdt[i] = gamma * I[i]

    return np.concatenate([dSdt, dIdt, dRdt])


# Parameters
num_communities = 3
population = 1000  # Population per community
total_population = population * num_communities
I0 = [
    1,
    0,
    0,
]  # Initial infected in each community (only first community has infection)
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
travel_prob = 0.05  # Probability of travel between communities per day

# Time points
t = np.linspace(0, 100, 1000)

# Initial conditions
S0 = [population - I0[i] for i in range(num_communities)]  # Susceptible per community
R0 = [0] * num_communities  # Recovered per community
initial_conditions = np.array(S0 + I0 + R0) / population  # Normalized

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

# Plotting
plt.figure(figsize=(12, 8))

for i in range(num_communities):
    plt.subplot(num_communities, 1, i + 1)
    plt.plot(t, S[:, i], "b", label="Susceptible")
    plt.plot(t, I[:, i], "r", label="Infected")
    plt.plot(t, R[:, i], "g", label="Recovered")
    plt.title(f"Community {i+1}")
    plt.ylabel("Number of Individuals")
    if i == num_communities - 1:
        plt.xlabel("Time (days)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()

# Add overall statistics
plt.figure(figsize=(10, 6))
plt.plot(t, np.sum(S, axis=1), "b", label="Total Susceptible")
plt.plot(t, np.sum(I, axis=1), "r", label="Total Infected")
plt.plot(t, np.sum(R, axis=1), "g", label="Total Recovered")
plt.title("Total Population Across All Communities")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid(True)

# Display parameters
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

plt.show()

# Print statistics
for i in range(num_communities):
    peak_infected = max(I[:, i])
    peak_day = t[np.argmax(I[:, i])]
    print(f"\nCommunity {i+1}:")
    print(f"Peak infection: {peak_infected:.0f} individuals")
    print(f"Day of peak infection: {peak_day:.1f}")
    print(f"Final recovered: {R[-1, i]:.0f} individuals")
