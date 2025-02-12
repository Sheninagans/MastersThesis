import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from thesis_code.analysis.algo4 import fit

def simulate_state_sequence(T, P):
    """Simulates a state sequence using the given transition probability matrix."""
    num_states = P.shape[0]
    states = np.zeros(T, dtype=int)
    states[0] = np.random.choice(num_states)  # Start from a random state
    for t in range(1, T):
        states[t] = np.random.choice(num_states, p=P[states[t-1]])
    return states

def generate_observations(states, mu, sigma):
    """Generates observations based on the state sequence."""
    T = len(states)
    Y = np.array([np.random.normal(mu[s], sigma[s]) for s in states])
    return Y.reshape(-1, 1)  # Ensure Y is a column vector

def apply_time_scaling(mu, sigma, P, scale):
    """Scales parameters based on time frame (daily, weekly, monthly)."""
    P_scaled = np.linalg.matrix_power(P, scale)
    P_scaled /= P_scaled.sum(axis=1, keepdims=True)  # Ensure transition probabilities sum to 1
    mu_scaled = mu * scale
    sigma_scaled = sigma * np.sqrt(scale)
    return mu_scaled, sigma_scaled, P_scaled

# Simulation parameters
T = 1000  # Number of time steps
num_states = 2
mu = np.array([0.0123, -0.0157])  # Mean returns per state
sigma = np.array([0.0347, 0.0778])  # Volatility per state
P = np.array([[0.9629, 0.0371], [0.2102, 0.7898]])  # Transition matrix




# Run simulation for different time scales
time_scales = {"Daily": 1}
penalties = {"Daily": 0.01}
#time_scales = {"Daily": 1, "Weekly": 5, "Monthly": 20}
#penalties = {"Daily": 100, "Weekly": 50, "Monthly": 1}

results = {}
for scale_name, scale_value in time_scales.items():
    print(f"Running simulation for {scale_name} data...")
    mu_scaled, sigma_scaled, P_scaled = apply_time_scaling(mu, sigma, P, scale_value)
    states = simulate_state_sequence(T, P_scaled)
    Y = generate_observations(states, mu_scaled, sigma_scaled)
    lambda_penalty = penalties[scale_name]
    theta_est, S_est = fit(Y, num_states=num_states, lambda_penalty=lambda_penalty, grid_size=0.05)
    
    results[scale_name] = (states, np.argmax(S_est, axis=1))

    # Compute estimated means and volatilities
    mu_est = np.mean(theta_est, axis=1)
    sigma_est = np.array([
        np.std(Y[np.argmax(S_est, axis=1) == k]) if np.sum(np.argmax(S_est, axis=1) == k) > 1 else np.nan
        for k in range(num_states)
    ])

    # Compute standard error of mu
    state_counts = np.array([np.sum(np.argmax(S_est, axis=1) == k) for k in range(num_states)])
    mu_se = sigma_est / np.sqrt(state_counts)  # Standard error of the mean

    # Compute Balanced Accuracy
    true_states = states  # Ground truth states from simulation
    predicted_states = np.argmax(S_est, axis=1)  # Estimated states from the model
    BAC = balanced_accuracy_score(true_states, predicted_states)
    
    # Print results
    print(f"{scale_name} - State Classification Accuracy (BAC): {BAC:.2%}")
    print(f"True means: {mu_scaled}")
    print(f"Estimated means: {mu_est}")
    print(f"Estimated standard errors (mu): {mu_se}")  # Print standard errors
    print(f"True volatilities: {sigma_scaled}")
    print(f"Estimated volatilities: {sigma_est}")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.plot(states, label="True States", linestyle='dashed')
    plt.plot(np.argmax(S_est, axis=1), label="Estimated States", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title(f"True vs. Estimated States ({scale_name})")
    plt.legend()
    plt.show()

