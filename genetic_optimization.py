"""
Genetic Algorithm for Optimizing Community Parameters in Virus Transmission Simulation

This module implements a constrained genetic algorithm to optimize community parameters
for minimizing virus transmission while respecting resource constraints. The optimization
considers both cultural and policy parameters that affect disease spread.

Constraints:
1. Total policy implementation cost cannot exceed budget
2. Cultural parameters must maintain minimum community cohesion
3. Policy parameters must maintain minimum effectiveness
4. Travel restrictions must be balanced between communities

The algorithm uses tournament selection, uniform crossover, and mutation with
constraint repair to maintain feasible solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from main import multi_community_sir
import streamlit as st
import random


class Individual:
    def __init__(self, num_communities):
        # Cultural parameters (0-1)
        self.culture = {
            "rule_following": np.random.random(
                num_communities
            ),  # Probability of following rules
            "community_cohesion": np.random.random(
                num_communities
            ),  # How well community works together
            "risk_perception": np.random.random(
                num_communities
            ),  # How seriously they take the threat
        }

        # Policy parameters (0-1)
        self.policies = {
            "quarantine_strictness": np.random.random(num_communities),
            "social_distancing_enforcement": np.random.random(num_communities),
            "testing_capacity": np.random.random(num_communities),
            "travel_restrictions": np.random.random(num_communities),
        }

        self.fitness = None
        self.constraint_violation = 0

    def get_community_params(self, t):
        # Time-dependent parameters
        vaccination_available = t >= 90  # After 3 months

        community_params = []
        for i in range(len(self.culture["rule_following"])):
            # Calculate effective parameters based on culture and policies
            quarantine_effectiveness = (
                self.policies["quarantine_strictness"][i]
                * self.culture["rule_following"][i]
                * self.culture["risk_perception"][i]
            )

            social_distancing = (
                self.policies["social_distancing_enforcement"][i]
                * self.culture["rule_following"][i]
                * self.culture["community_cohesion"][i]
            )

            testing_rate = (
                self.policies["testing_capacity"][i]
                * self.culture["risk_perception"][i]
            )

            # Vaccination only available after 3 months
            vaccination_rate = (
                self.culture["rule_following"][i]
                * self.culture["risk_perception"][i]
                * vaccination_available
            )

            travel_restrictions = (
                self.policies["travel_restrictions"][i]
                * self.culture["rule_following"][i]
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

        return community_params

    def check_constraints(self):
        """Check if the individual's parameters satisfy all constraints."""
        violation = 0

        # Constraint 1: Total policy implementation cost
        total_cost = sum(sum(policy) for policy in self.policies.values())
        if total_cost > self.max_budget:
            violation += (total_cost - self.max_budget) ** 2

        # Constraint 2: Minimum community cohesion
        min_cohesion = 0.3
        for cohesion in self.culture["community_cohesion"]:
            if cohesion < min_cohesion:
                violation += (min_cohesion - cohesion) ** 2

        # Constraint 3: Minimum policy effectiveness
        min_effectiveness = 0.2
        for policy in self.policies.values():
            for value in policy:
                if value < min_effectiveness:
                    violation += (min_effectiveness - value) ** 2

        # Constraint 4: Balanced travel restrictions
        travel_restrictions = self.policies["travel_restrictions"]
        mean_restriction = np.mean(travel_restrictions)
        for restriction in travel_restrictions:
            if abs(restriction - mean_restriction) > 0.3:
                violation += (abs(restriction - mean_restriction) - 0.3) ** 2

        self.constraint_violation = violation
        return violation == 0

    def repair_constraints(self):
        """Repair constraint violations while maintaining solution quality."""
        # Repair total cost constraint
        total_cost = sum(sum(policy) for policy in self.policies.values())
        if total_cost > self.max_budget:
            scale_factor = self.max_budget / total_cost
            for policy in self.policies.values():
                policy *= scale_factor

        # Repair minimum community cohesion
        min_cohesion = 0.3
        for i in range(len(self.culture["community_cohesion"])):
            if self.culture["community_cohesion"][i] < min_cohesion:
                self.culture["community_cohesion"][i] = min_cohesion

        # Repair minimum policy effectiveness
        min_effectiveness = 0.2
        for policy in self.policies.values():
            for i in range(len(policy)):
                if policy[i] < min_effectiveness:
                    policy[i] = min_effectiveness

        # Repair travel restrictions balance
        travel_restrictions = self.policies["travel_restrictions"]
        mean_restriction = np.mean(travel_restrictions)
        for i in range(len(travel_restrictions)):
            if abs(travel_restrictions[i] - mean_restriction) > 0.3:
                travel_restrictions[i] = mean_restriction + 0.3 * np.sign(
                    travel_restrictions[i] - mean_restriction
                )

    def mutate(self, mutation_rate=0.1):
        # Mutate cultural parameters
        for param in self.culture:
            for i in range(len(self.culture[param])):
                if random.random() < mutation_rate:
                    self.culture[param][i] = np.clip(
                        self.culture[param][i] + random.gauss(0, 0.1), 0, 1
                    )

        # Mutate policy parameters
        for param in self.policies:
            for i in range(len(self.policies[param])):
                if random.random() < mutation_rate:
                    self.policies[param][i] = np.clip(
                        self.policies[param][i] + random.gauss(0, 0.1), 0, 1
                    )

        # Repair any constraint violations after mutation
        self.repair_constraints()


class GeneticAlgorithm:
    def __init__(
        self, population_size=20, num_generations=50, num_communities=3, max_budget=10
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_communities = num_communities
        self.max_budget = max_budget
        self.population = [Individual(num_communities) for _ in range(population_size)]
        for ind in self.population:
            ind.max_budget = max_budget
            ind.repair_constraints()
        self.best_individual = None
        self.best_fitness = float("inf")
        self.fitness_history = []

    def evaluate_fitness(self, individual):
        # Check constraints first
        if not individual.check_constraints():
            return float("inf")  # Penalize constraint violations

        # Simulation parameters
        beta = 0.3
        gamma = 0.1
        travel_prob = 0.05
        population = 1000

        # Time points
        t = np.linspace(0, 180, 1000)  # 6 months simulation

        # Initial conditions
        S0 = [population - 1 for _ in range(self.num_communities)]
        I0 = [1] + [0] * (self.num_communities - 1)
        R0 = [0] * self.num_communities
        initial_conditions = np.array(S0 + I0 + R0) / population

        # Get time-dependent parameters
        community_params = individual.get_community_params(
            t[0]
        )  # Use initial time for parameters

        # Solve ODE
        solution = odeint(
            multi_community_sir,
            initial_conditions,
            t,
            args=(
                beta,
                gamma,
                self.num_communities,
                travel_prob,
                population,
                community_params,
            ),
        )

        # Extract results
        I = solution[:, self.num_communities : 2 * self.num_communities] * population

        # Calculate fitness (lower is better)
        # Consider peak infections, total infections, and duration of outbreak
        peak_infections = np.max(I)
        total_infections = np.sum(I)
        outbreak_duration = len(np.where(I > 1)[0]) / len(t)

        # Weighted sum of different metrics
        fitness = (
            0.5 * peak_infections
            + 0.3 * total_infections
            + 0.2 * outbreak_duration * population
        )

        return fitness

    def crossover(self, parent1, parent2):
        child = Individual(self.num_communities)
        child.max_budget = self.max_budget

        # Crossover cultural parameters
        for param in child.culture:
            for i in range(len(child.culture[param])):
                if random.random() < 0.5:
                    child.culture[param][i] = parent1.culture[param][i]
                else:
                    child.culture[param][i] = parent2.culture[param][i]

        # Crossover policy parameters
        for param in child.policies:
            for i in range(len(child.policies[param])):
                if random.random() < 0.5:
                    child.policies[param][i] = parent1.policies[param][i]
                else:
                    child.policies[param][i] = parent2.policies[param][i]

        # Repair any constraint violations after crossover
        child.repair_constraints()

        return child

    def evolve(self, progress_bar):
        for generation in range(self.num_generations):
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual)

            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness)

            # Update best individual
            if self.population[0].fitness < self.best_fitness:
                self.best_individual = self.population[0]
                self.best_fitness = self.population[0].fitness

            # Record fitness history
            self.fitness_history.append(self.best_fitness)

            # Create new population
            new_population = [self.best_individual]  # Elitism

            # Tournament selection and crossover
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament = random.sample(self.population, tournament_size)
                parent1 = min(tournament, key=lambda x: x.fitness)

                tournament = random.sample(self.population, tournament_size)
                parent2 = min(tournament, key=lambda x: x.fitness)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child.mutate()

                new_population.append(child)

            self.population = new_population

            # Update progress bar
            progress = (generation + 1) / self.num_generations
            progress_bar.progress(progress)


def run_genetic_algorithm():
    st.title("Constrained Genetic Algorithm Optimization of Community Parameters")

    st.write(
        """
    This optimization considers the following constraints:
    1. Total policy implementation cost cannot exceed budget
    2. Cultural parameters must maintain minimum community cohesion
    3. Policy parameters must maintain minimum effectiveness
    4. Travel restrictions must be balanced between communities
    """
    )

    # Parameters
    population_size = st.sidebar.slider("Population Size", 10, 50, 20)
    num_generations = st.sidebar.slider("Number of Generations", 10, 100, 50)
    num_communities = st.sidebar.slider("Number of Communities", 2, 5, 3)
    max_budget = st.sidebar.slider("Maximum Budget", 5.0, 20.0, 10.0)

    if st.button("Run Optimization"):
        # Create progress bar
        progress_bar = st.progress(0)

        # Initialize and run genetic algorithm
        ga = GeneticAlgorithm(
            population_size, num_generations, num_communities, max_budget
        )
        ga.evolve(progress_bar)

        # Clear progress bar
        progress_bar.empty()

        # Plot fitness history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ga.fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness (Lower is Better)")
        ax.set_title("Evolution of Best Fitness")
        st.pyplot(fig)

        # Display best individual's parameters
        st.subheader("Best Individual's Parameters")

        # Cultural parameters
        st.write("### Cultural Parameters")
        for param in ga.best_individual.culture:
            st.write(f"**{param.replace('_', ' ').title()}:**")
            for i, value in enumerate(ga.best_individual.culture[param]):
                st.write(f"Community {i+1}: {value:.3f}")

        # Policy parameters
        st.write("### Policy Parameters")
        for param in ga.best_individual.policies:
            st.write(f"**{param.replace('_', ' ').title()}:**")
            for i, value in enumerate(ga.best_individual.policies[param]):
                st.write(f"Community {i+1}: {value:.3f}")

        # Display constraint satisfaction
        st.write("### Constraint Satisfaction")
        total_cost = sum(sum(policy) for policy in ga.best_individual.policies.values())
        st.write(f"Total Policy Cost: {total_cost:.2f} (Budget: {max_budget:.2f})")

        min_cohesion = min(ga.best_individual.culture["community_cohesion"])
        st.write(f"Minimum Community Cohesion: {min_cohesion:.2f} (Required: 0.30)")

        min_effectiveness = min(
            min(policy) for policy in ga.best_individual.policies.values()
        )
        st.write(
            f"Minimum Policy Effectiveness: {min_effectiveness:.2f} (Required: 0.20)"
        )

        travel_restrictions = ga.best_individual.policies["travel_restrictions"]
        max_diff = max(
            abs(r - np.mean(travel_restrictions)) for r in travel_restrictions
        )
        st.write(f"Maximum Travel Restriction Difference: {max_diff:.2f} (Limit: 0.30)")

        # Run final simulation with best parameters
        st.subheader("Final Simulation Results")

        # Simulation parameters
        beta = 0.3
        gamma = 0.1
        travel_prob = 0.05
        population = 1000

        # Time points
        t = np.linspace(0, 180, 1000)

        # Initial conditions
        S0 = [population - 1 for _ in range(num_communities)]
        I0 = [1] + [0] * (num_communities - 1)
        R0 = [0] * num_communities
        initial_conditions = np.array(S0 + I0 + R0) / population

        # Get time-dependent parameters
        community_params = ga.best_individual.get_community_params(
            t[0]
        )  # Use initial time for parameters

        # Solve ODE
        solution = odeint(
            multi_community_sir,
            initial_conditions,
            t,
            args=(
                beta,
                gamma,
                num_communities,
                travel_prob,
                population,
                community_params,
            ),
        )

        # Extract results
        S = solution[:, :num_communities] * population
        I = solution[:, num_communities : 2 * num_communities] * population
        R = solution[:, 2 * num_communities :] * population

        # Plot results
        fig2, axes = plt.subplots(num_communities, 1, figsize=(10, 4 * num_communities))
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
        st.pyplot(fig2)


if __name__ == "__main__":
    run_genetic_algorithm()
