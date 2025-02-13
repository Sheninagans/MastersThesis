import numpy as np
import os
import sys
# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from thesis_code.analysis.algo4 import fit_parameters

# Generate example data
np.random.seed(42)  # For reproducibility
Y = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])  # 4 samples, 2 features

# Example soft state assignment matrix (S) with 2 regimes (K=2)
S = np.array([[0.8, 0.2],
              [0.6, 0.4],
              [0.3, 0.7],
              [0.1, 0.9]])  # 4 samples, 2 states

K = S.shape[1]  # Number of regimes

# Function call
theta = fit_parameters(Y, S, K)

# Output results
print("Estimated Parameters for Each Regime:")
print(theta)
