from thesis_code.utils.helpers import export_to_excel
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linprog

# Ensure scipy is installed correctly
try:
    from scipy.optimize import linprog
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "scipy"])
    from scipy.optimize import linprog

def initialize_states(Y, K):
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10)
    return np.eye(K)[kmeans.fit_predict(Y)]

def fit_parameters(Y, S, K):
    return np.array([
        np.average(Y, axis=0, weights=S[:, k]) for k in range(K)
    ])

def fit_state_sequence(Y, theta, K, lambda_penalty):
    T = Y.shape[0]
    L = np.array([[np.linalg.norm(Y[t] - theta[k])**2 for k in range(K)] for t in range(T)])
    
    # Convert to linear programming problem
    c = L.flatten()
    A_eq = np.tile(np.eye(T), (1, K))
    b_eq = np.ones(T)
    bounds = [(0, 1)] * (T * K)
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.x.reshape(T, K) if res.success else None

def fit(Y, num_states, lambda_penalty, max_iter=10, tol=1e-6):
    K = num_states
    S = initialize_states(Y, K)
    
    for _ in range(max_iter):
        prev_S = S.copy()
        theta = fit_parameters(Y, S, K)
        S = fit_state_sequence(Y, theta, K, lambda_penalty)
        if S is None or np.linalg.norm(S - prev_S) < tol:
            break
    
    return theta, S
