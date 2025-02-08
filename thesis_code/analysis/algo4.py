from thesis_code.utils.helpers import export_to_excel
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cvxpy as cp

# Ensure scipy is installed correctly
try:
    from scipy.optimize import linprog
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "scipy"])
    from scipy.optimize import linprog


def fit_parameters(Y, S, K):
    """
    Compute the model parameters Theta given the state assignments S.
    """
    return np.array([
        np.average(Y, axis=0, weights=S[:, k]) for k in range(K)
    ])


def fit_state_sequence(Y, theta, C, lambda_penalty, num_states):
    """
    Algorithm 5: Solve for the state sequence S using dynamic programming.
    """
    T, N = Y.shape[0], C.shape[1]
    Y_array = Y.to_numpy()  # Convert DataFrame to array
    L = np.array([[np.linalg.norm(Y_array[t] - theta.T @ C[:, i])**2 for i in range(N)] for t in range(T)])
    
    # Compute jump penalty matrix
    Lambda = lambda_penalty / 4 * np.sum(np.abs(C[:, :, None] - C[:, None, :]), axis=0)
    
    # Initialize DP table
    V = np.zeros((T, N))
    V[0, :] = L[0, :]
    
    for t in range(1, T):
        for i in range(N):
            V[t, i] = L[t, i] + np.min(V[t-1, :] + Lambda[:, i])
    
    # Backtrack to retrieve optimal state sequence
    S_opt = np.zeros((T, num_states))  # Ensure shape matches `S`
    idx = np.argmin(V[-1, :])
    for t in range(T-1, -1, -1):
        S_opt[t, idx] = 1
        idx = np.argmin(V[t-1, :] + Lambda[:, idx])
    
    return S_opt


def fit(Y, num_states, lambda_penalty, grid_size=0.05):
    """
    Algorithm 4: Coordinate descent algorithm to fit the continuous jump model.
    """
    T, D = Y.shape
    
    # Initialize clusters with K-means++
    kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=10).fit(Y)
    theta = kmeans.cluster_centers_
    
    # Discretize probability simplex
    C = np.linspace(0, 1, num_states).reshape(1, -1)
    C = np.vstack([C, 1 - C])  # Ensure `C` has exactly `num_states` rows
    
    # Initialize S randomly
    S = np.random.dirichlet(np.ones(num_states), size=T)
    
    for _ in range(10):
        print("✅ Debug: S sum per column:", S.sum(axis=0))
        
        theta = fit_parameters(Y, S, num_states)  # Step (a): Update Theta
        S_new = fit_state_sequence(Y, theta, C, lambda_penalty, num_states)  # Step (b): Call Algorithm 5
        
        print("✅ Debug: S shape:", S.shape)
        print("✅ Debug: S_new shape:", S_new.shape)
        
        if np.allclose(S, S_new):
            break
        
        S = S_new  # Update S after convergence check
        
        # Ensure no column in S is entirely zero
        S += 1e-8  # Small constant to prevent zero values
        S /= S.sum(axis=1, keepdims=True)  # Normalize each row to sum to 1
    
    print("✅ Debug: S shape before returning:", S.shape)  # Fixed from S_opt.shape
    return theta, S
