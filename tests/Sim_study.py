import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import gc
import time
from scipy.linalg import expm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from thesis_code.analysis.algo4 import fit
from thesis_code.analysis.algo1 import generate_features
from thesis_code.utils.helpers import export_to_excel

def simulate_continuous_markov_returns(T, P, mu, sigma):
    """Simulates a return process with a discrete-time Markov chain and GBM-style normal shocks."""
    num_states = P.shape[0]
    states = np.zeros(T, dtype=int)
    states[0] = np.random.choice(num_states, p=[0.75, 0.25])  # Stationary distribution approximation
    returns = np.zeros(T)

    mu_annual = mu  
    sigma_annual = sigma  # Annualize monthly volatilities
    dt = 1  # Daily 

    # Ensure P is correctly normalized
    P = P / P.sum(axis=1, keepdims=True)

    for t in range(1, T):
        states[t] = np.random.choice(num_states, p=P[states[t - 1]])
        dW = np.random.normal(0, np.sqrt(dt))  
        returns[t] = mu_annual[states[t]] * dt + sigma_annual[states[t]] * dW

    return states, returns

def compute_regime_statistics(Y, S, true_states):
    """Computes accuracy, mean returns, volatilities, jump probability, and classification metrics."""
    num_states = S.shape[1]
    min_length = min(len(Y), len(S), len(true_states))
    Y, S, true_states = Y[:min_length], S[:min_length], true_states[:min_length]
    
    # Assign predicted states (Hard Classification)
    pred_states = np.argmax(S, axis=1)
    
    # Compute Mean & Volatility Estimates
    mu_est = np.array([np.mean(Y[pred_states == k]) if np.any(pred_states == k) else np.nan for k in range(num_states)])
    sigma_est = np.array([np.std(Y[pred_states == k], ddof=1) if np.any(pred_states == k) else np.nan for k in range(num_states)])
    
    # Jump Probability
    num_transitions = np.sum(pred_states[1:] != pred_states[:-1])
    gamma = num_transitions / (len(pred_states) - 1)

    # Directional Jump Probabilities (New Addition)
    transitions = np.zeros((num_states, num_states))
    state_counts = np.zeros(num_states)

    for t in range(len(pred_states) - 1):
        transitions[pred_states[t], pred_states[t+1]] += 1
        state_counts[pred_states[t]] += 1

    gamma_12 = transitions[0, 1] / state_counts[0] if state_counts[0] > 0 else np.nan
    gamma_21 = transitions[1, 0] / state_counts[1] if state_counts[1] > 0 else np.nan
    
    # Compute Classification Metrics
    cm = confusion_matrix(true_states, pred_states, labels=np.arange(num_states))
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    bac = balanced_accuracy_score(true_states, pred_states)
    roc_auc = roc_auc_score(true_states, S[:, 1]) if num_states == 2 else np.nan

    return mu_est, sigma_est, gamma, gamma_12, gamma_21, accuracy_per_class, bac, roc_auc, pred_states

def run_simulation(T, P, mu, sigma, num_states=2, lambda_penalty=1000):
    """Runs a single simulation with properly aligned returns and feature set."""
    states, returns = simulate_continuous_markov_returns(T, P, mu, sigma)
    X = generate_features(returns, window_lengths=[6, 14])
    returns = returns[13:]
    theta_est, S_est = fit(X, num_states=num_states, lambda_penalty=lambda_penalty, grid_size=0.02, tolerance=1e-7)

    return compute_regime_statistics(returns, S_est, states[13:]), states


def run_simulation_wrapper(i, T, P, mu, sigma, num_states, lambda_penalty):
    """Wrapper for parallel execution to maintain consistent argument passing."""
    result = run_simulation(T, P, mu, sigma, num_states=num_states, lambda_penalty=lambda_penalty)
    gc.collect()  # Clean up memory after each run
    return result

def parallel_simulation(num_simulations, T, P, mu, sigma, num_states=2, lambda_penalty=1000):
    """Runs multiple simulations in parallel while reducing overhead."""
    n_jobs = min(4, multiprocessing.cpu_count() - 1)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size=2)(
        delayed(run_simulation_wrapper)(i, T, P, mu, sigma, num_states, lambda_penalty)
        for i in range(num_simulations)
    )

    # Unpack results
    stats, state_sequences = zip(*results)
    mu_vals, sigma_vals, gamma_vals, gamma_12_vals, gamma_21_vals, acc_per_class_vals, bac_vals, roc_auc_vals, _ = zip(*stats)

    # Compute summary statistics
    summary = {
        "mu_mean": np.nanmean(mu_vals, axis=0), "mu_std": np.nanstd(mu_vals, axis=0),
        "sigma_mean": np.nanmean(sigma_vals, axis=0), "sigma_std": np.nanstd(sigma_vals, axis=0),
        "gamma_mean": np.nanmean(gamma_vals), "gamma_std": np.nanstd(gamma_vals),
        "gamma_12_mean": np.nanmean(gamma_12_vals), "gamma_12_std": np.nanstd(gamma_12_vals),
        "gamma_21_mean": np.nanmean(gamma_21_vals), "gamma_21_std": np.nanstd(gamma_21_vals),
        "accuracy_per_class_mean": np.nanmean(acc_per_class_vals, axis=0),
        "accuracy_per_class_std": np.nanstd(acc_per_class_vals, axis=0),
        "bac_mean": np.nanmean(bac_vals), "bac_std": np.nanstd(bac_vals),
        "roc_auc_mean": np.nanmean(roc_auc_vals), "roc_auc_std": np.nanstd(roc_auc_vals),
        "state_sequences": state_sequences
    }

    gc.collect()
    return summary


T = 1000  # Number of months (instead of days)
P = np.array([[0.9979, 0.0021], [0.0120, 0.9880]])  # Transition probability matrix (Daily)
mu = np.array([0.000615, -0.000785])  # Daily mean returns
sigma = np.array([0.007759, 0.0174])  # Daily volatilities

start_time = time.time()
end_time = time.time()

num_simulations = 1

summary_stats = parallel_simulation(num_simulations, T, P, mu, sigma)

# Display summary statistics
# Print summary
print("\nParallel Simulation Summary:")
print("Estimated Mean Returns (μ):", summary_stats["mu_mean"], "±", summary_stats["mu_std"])
print("Estimated Volatilities (σ):", summary_stats["sigma_mean"], "±", summary_stats["sigma_std"])
print("Overall Jump Probability (γ):", summary_stats["gamma_mean"], "±", summary_stats["gamma_std"])
print("Jump Probability 1 → 2 (γ_12):", summary_stats["gamma_12_mean"], "±", summary_stats["gamma_12_std"])
print("Jump Probability 2 → 1 (γ_21):", summary_stats["gamma_21_mean"], "±", summary_stats["gamma_21_std"])
print("Accuracy per class:", summary_stats["accuracy_per_class_mean"], "±", summary_stats["accuracy_per_class_std"])
print("Balanced Accuracy (BAC):", summary_stats["bac_mean"], "±", summary_stats["bac_std"])
print("ROC-AUC:", summary_stats["roc_auc_mean"], "±", summary_stats["roc_auc_std"])
