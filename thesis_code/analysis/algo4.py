from thesis_code.utils.helpers import export_to_excel
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cvxpy as cp

def fit_parameters(Y, S, K):
    theta = []
    
    for k in range(K):
        #print("✅ Debug: S[:, k].shape:", S[:, k].shape)
        #print("✅ Debug: Y.shape:", Y.shape)
        #print("✅ Debug: Weighted sum:", (np.sum(S[:, k, None] * Y)))
        #print("✅ Debug: Denominator sum:", np.sum(S[:, k]))
        
        theta_k = np.sum(S[:, k, None] * Y, axis=0) / (np.sum(S[:, k]) + 1e-8)
        theta.append(theta_k)
    
    return np.array(theta)


def fit_state_sequence(Y, theta, C, lambda_penalty, num_states):
    """
    Algorithm 5: Solve for the state sequence S using dynamic programming.
    """
    T, N = Y.shape[0], C.shape[1]
    Y_array = Y if isinstance(Y, np.ndarray) else Y.to_numpy() # Convert DataFrame to array if nessecary
    
    L_raw = np.array([
    [0.5*np.sum((Y_array[t] - theta[k, :])**2)
    for k in range(theta.shape[0])]
    for t in range(Y_array.shape[0])])
    #L = np.sum(L_raw[:, :, None] * C[:, None, :], axis=1)
    L = L_raw @ C  # Apply C transformation afterward
    #print("L", L)
    #export_to_excel(L,filename="L.csv")
    
    #L = np.array([[np.linalg.norm(Y_array[t] - theta.T @ C[:, i])**2 for i in range(N)] for t in range(T)])
    
    
    
    # Compute jump penalty matrix
    Lambda = np.round(lambda_penalty / 4 * np.sum(np.abs(C[:, :, None] - C[:, None, :])**2, axis=0), decimals=10)
        # Initialize DP table
    V = np.zeros((T, N))
    V[0, :] = L[0, :]

    backtrack = np.zeros((T, N), dtype=int)  # Store previous states for backtracking
    for t in range(1, T):
        for i in range(N):
            prev_state = np.argmin(V[t-1, :])
            backtrack[t, i] = prev_state  # Store best previous state
            #print("Prev state:", prev_state,"Lambda[prev_state]", Lambda[prev_state, i])
            #print("V[t-1, prev_state]:", V[t-1, prev_state])
            V[t, i] = L[t, i] + V[t-1, prev_state] + Lambda[prev_state, i]
            #print("V:", V)

    #print("✅ Debug: V shape:", V.shape)
    #export_to_excel(V,filename="V.csv")

        # Backtrack to retrieve optimal state sequence
    S_opt = np.zeros((T, num_states))
    idx = np.argmin(V[-1, :])
    for t in range(T-1, -1, -1):
        S_opt[t] = C[:, idx]
        if t > 0:
            idx = backtrack[t, idx]  # Use stored best previous state
        
        
        #print("Final V values used for backtracking:", np.argmin(V[t-1, :]+Lambda[:, idx]))
        #print("Penalty matrix V:", V[t-1, :]+Lambda[:, idx])
        #print("Penalty matrix C:", C[:, idx])
        #print(t)
    #print("Loss matrix L:")
    #print(L)
    #print("Penalty matrix Lambda:")
    #print(Lambda)
    return S_opt


def fit(Y, num_states, lambda_penalty, grid_size):
    """
    Algorithm 4: Coordinate descent algorithm to fit the continuous jump model.
    """
    T, D = Y.shape
    
    # Initialize clusters with K-means++
    kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=10).fit(Y)
    theta = kmeans.cluster_centers_
    
    # Discretize probability simplex
    # Create a probability grid with `grid_size` resolution
    C_vals = np.arange(0, 1 + grid_size, grid_size)  # Discretized values from 0 to 1
    C = np.array(np.meshgrid(*([C_vals] * num_states))).T.reshape(-1, num_states).T
    C = C[:, np.isclose(C.sum(axis=0), 1)]
     #print("✅ Debug: C shape:", C.shape)   
    #print(C)
    export_to_excel(C,filename="C.csv")
    
    # Initialize S randomly
    S = np.random.dirichlet(np.ones(num_states), size=T)

        
    for iter_counter in range(100):  # Larger iteration limit
        theta = fit_parameters(Y, S, num_states)
        S_new = fit_state_sequence(Y, theta, C, lambda_penalty, num_states)
        #print(f"Iteration {iter_counter}: Change in S = {np.linalg.norm(S - S_new)}")
        if iter_counter>5 and np.linalg.norm(S - S_new) < 1e-6:
            break
        S = S_new
        
        # Ensure no column in S is entirely zero
       # S += 1e-8  # Small constant to prevent zero values
       # S /= S.sum(axis=1, keepdims=True)  # Normalize each row to sum to 1
    
    #print("✅ Debug: S shape before returning:", S.shape)  # Fixed from S_opt.shape
    export_to_excel(S,filename="S.csv")
    export_to_excel(C,filename="C.csv")
    export_to_excel(theta,filename="theta.csv")
    return theta, S
