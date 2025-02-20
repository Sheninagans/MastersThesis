from thesis_code.utils.helpers import export_to_excel
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cvxpy as cp
import random


def fit_parameters(Y, S, K):
    theta = []
    for k in range(K):
        theta_k = np.sum(S[:, k, None] * Y, axis=0) / (np.sum(S[:, k]) + 1e-8)
        theta.append(theta_k)
    return np.array(theta)

def fit_state_sequence(Y, theta, C, lambda_penalty, num_states):
    T, N = Y.shape[0], C.shape[1]
    Y_array = Y if isinstance(Y, np.ndarray) else Y.to_numpy()
    
    L_raw = np.array([
        [0.5 * np.sum((Y_array[t] - theta[k, :])**2) for k in range(theta.shape[0])]
        for t in range(Y_array.shape[0])])
    
    L = L_raw @ C
    Lambda = np.round(lambda_penalty / 4 * np.sum(np.abs(C[:, :, None] - C[:, None, :]), axis=0)**2, decimals=10)
    
    V = np.zeros((T, N))
    V[0, :] = L[0, :]
        
    backtrack = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        for i in range(N):
            prev_state = np.argmin(V[t-1, :] + Lambda[:, i])
            V[t, i] = L[t, i] + V[t-1, prev_state] + Lambda[prev_state, i]
    
        S_opt = np.zeros((T, num_states))
    idx = np.argmin(V[-1, :])
    for t in range(T-1, -1, -1):
        S_opt[t] = C[:, idx]
        if t > 0:
            idx = np.argmin(V[t-1, :]+Lambda[:,idx])
        
    return S_opt


def initialize_S_kmeans(Y, num_states):
    """
    Initializes the state matrix S using K-means++ clustering with soft assignments.

    Parameters:
    Y (numpy.ndarray or pd.DataFrame): The feature matrix (T x D).
    num_states (int): The number of states/clusters.

    Returns:
    S (numpy.ndarray): The initialized state matrix (T x num_states) with soft assignments.
    """

    # Ensure Y is a NumPy array
    Y_array = Y if isinstance(Y, np.ndarray) else Y.to_numpy()

    # Apply K-means++ clustering
    kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(Y_array)

    # Compute squared Euclidean distances from each point to each cluster center
    distances = np.linalg.norm(Y_array[:, None, :] - kmeans.cluster_centers_, axis=2)  # Shape: (T, num_states)

    # Convert distances to similarity scores (higher similarity = closer to cluster center)
    similarity_scores = np.exp(-distances)  # Inverse distance weighting

    # Normalize to ensure each row sums to 1 (soft clustering)
    S = similarity_scores / similarity_scores.sum(axis=1, keepdims=True)

    return S




def fit(Y, num_states, lambda_penalty, grid_size,tolerance):
    T, D = Y.shape
    
    # Discretize probability simplex
    C_vals = np.arange(0, 1 + grid_size, grid_size)
    C = np.array(np.meshgrid(*([C_vals] * num_states))).T.reshape(-1, num_states).T
    C = C[:, np.isclose(C.sum(axis=0), 1)]

    export_to_excel(C, filename="C.csv")
    best_S = None
    best_theta = None
    best_obj_value = float('inf')
    
    obj_values = []
    S_initializations = []  # Store all initial S matrices

    
    
    for i in range(10):  # Run 10 times with different K-means++ initializations
        #sample_indices = np.random.choice(Y.shape[0], size=int(0.8 * Y.shape[0]), replace=False)  
        #Y_sample = Y.iloc[sample_indices]  # Subsample for centroid initialization  
        #Fit K-means on the full dataset but use initial centroids from Y_sample
        #kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=1, random_state=None).fit(Y)
        S=initialize_S_kmeans(Y, num_states)
        #S_initializations.append(S.copy())  # Save initial S for later analysis

        #S = np.zeros((T, num_states))
        #S[np.arange(T), kmeans.labels_] = 1  # One-hot encode initial states
  
        for iter_counter in range(100):
            theta = fit_parameters(Y, S, num_states)
            S_new = fit_state_sequence(Y, theta, C, lambda_penalty, num_states)
            if iter_counter > 5 and np.linalg.norm(S - S_new) < tolerance:
                break
            S = S_new
        
        obj_value = np.linalg.norm(S @ theta - Y)  # Objective function value
        obj_values.append(obj_value)
        
        if obj_value < best_obj_value:
            best_obj_value = obj_value
            best_theta = theta
            best_S = S
    
    S_df = pd.DataFrame(S_initializations)
    export_to_excel(S_df, filename="S_initializations.csv")  # Save for analysis


    export_to_excel(best_S, filename="S_Opt.csv")
    export_to_excel(best_theta, filename="theta_opt.csv")
    #export_to_excel(np.array(obj_values), filename="obj_values.csv")
    #export_to_excel(np.array(kmeans_results), filename="kmeans_results.csv")
    
    return best_theta, best_S

