import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.linalg import expm  # Needed for continuous-time Markov chain scaling

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from thesis_code.analysis.algo4 import fit
from thesis_code.analysis.algo1 import generate_features

def simulate_continuous_markov_returns(T, Q, mu, sigma, dt=1/252):
    """Simulates a GBM-compatible return process with a continuous-time Markov chain."""
    P_scaled = expm(Q * dt)  # Convert generator matrix Q to transition matrix P

    # Convert annual to daily values
    mu_daily = mu / 252
    sigma_daily = sigma / np.sqrt(252)

    num_states = Q.shape[0]
    states = np.zeros(T, dtype=int)
    states[0] = np.random.choice(num_states)
    returns = np.zeros(T)

    for t in range(1, T):
        states[t] = np.random.choice(num_states, p=P_scaled[states[t-1]])
        dW = np.random.normal(0, np.sqrt(dt))
        returns[t] = mu[states[t]]*dt + sigma_daily[states[t]] * dW   


    return states, returns

def compute_regime_statistics(Y, S):
    """Computes mean returns, volatilities, and sample jump probability (gamma)."""
    num_states = S.shape[1]
    min_length = min(len(Y), len(S))
    Y, S = Y[:min_length], S[:min_length]
    state_assignments = np.argmax(S, axis=1)
    
    mu_est = np.array([np.mean(Y[state_assignments == k]) if np.any(state_assignments == k) else np.nan for k in range(num_states)])
    sigma_est = np.array([np.std(Y[state_assignments == k], ddof=1) if np.any(state_assignments == k) else np.nan for k in range(num_states)])
    num_transitions = np.sum(state_assignments[1:] != state_assignments[:-1])
    gamma = num_transitions / (len(state_assignments) - 1)
    for k in range(num_states):
        regime_returns = Y[state_assignments == k]
        print(f"Regime {k}: {len(regime_returns)} observations")
    return mu_est, sigma_est, gamma, state_assignments

def estimate_transition_matrix(states, num_states=2):
    """Estimate empirical transition matrix from simulated states."""
    counts = np.zeros((num_states, num_states))
    for t in range(1, len(states)):
        counts[states[t - 1], states[t]] += 1
    return counts / counts.sum(axis=1, keepdims=True)

def run_simulation(T, Q, mu, sigma, num_states=2, dt=1/252, lambda_penalty=100):
    """Runs a single simulation with properly aligned returns and feature set."""
    states, returns = simulate_continuous_markov_returns(T, Q, mu, sigma, dt=dt)
    X = generate_features(returns, window_lengths=[6, 14])
    returns = returns[13:]
    theta_est, S_est = fit(X, num_states=num_states, lambda_penalty=lambda_penalty, grid_size=0.05, tolerance=1e-7)
    return compute_regime_statistics(returns, S_est), states

def parallel_simulation(num_simulations, T, Q, mu, sigma, num_states=2, dt=1/252, lambda_penalty=100, n_jobs=5):
    """Runs multiple simulations in parallel while preventing joblib deadlocks."""
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_simulation)(T, Q, mu, sigma, num_states, dt, lambda_penalty) for _ in range(num_simulations)
    )
    stats, state_sequences = zip(*results)
    mu_vals, sigma_vals, gamma_vals, _ = zip(*stats)
    return {
        "mu_mean": np.nanmean(mu_vals, axis=0), "mu_std": np.nanstd(mu_vals, axis=0),
        "sigma_mean": np.nanmean(sigma_vals, axis=0), "sigma_std": np.nanstd(sigma_vals, axis=0),
        "gamma_mean": np.nanmean(gamma_vals), "gamma_std": np.nanstd(gamma_vals),
        "state_sequences": state_sequences
    }

# Example Usage
T = 1000
#Q = np.array([[-0.0371, 0.0371], [0.2102, -0.2102]])  # Generator matrix

Q = np.array([[-0.37,  0.37], 
              [ 2.10, -2.10]])  # Increased magnitude

mu = np.array([0.0123, -0.0157])  # Annual mean returns
sigma = np.array([0.0347, 0.0778])  # Annual volatilities

# Run single simulation
(stats, states) = run_simulation(T, Q, mu, sigma)
mu_est, sigma_est, gamma_est, _ = stats
print("Single Simulation Results:")
print("Estimated Mean Returns:", mu_est)
print("Estimated Volatilities:", sigma_est)
print("Estimated Jump Probability:", gamma_est)

# Estimate empirical transition matrix
P_empirical = estimate_transition_matrix(states)
print("\nEmpirical Transition Matrix:")
print(P_empirical)

# Run parallel simulations
# num_simulations = 10
# summary_stats = parallel_simulation(num_simulations, T, Q, mu, sigma)
# print("\nParallel Simulation Summary:")
# print("Estimated Mean Returns (μ):", summary_stats["mu_mean"], "±", summary_stats["mu_std"])
# print("Estimated Volatilities (σ):", summary_stats["sigma_mean"], "±", summary_stats["sigma_std"])
# print("Estimated Jump Probability (γ):", summary_stats["gamma_mean"], "±", summary_stats["gamma_std"])

# # Estimate empirical transition matrix for the first state sequence
# P_empirical_parallel = estimate_transition_matrix(summary_stats["state_sequences"][0])
# print("\nEmpirical Transition Matrix from Parallel Simulations:")
# print(P_empirical_parallel)
