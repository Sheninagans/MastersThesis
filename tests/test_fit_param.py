import numpy as np
import os
import sys
from scipy.linalg import logm, expm
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

#  Output results
# #print("Estimated Parameters for Each Regime:")
# print(theta)

p_monthly = np.array([[0.9629, 0.0371], [0.2102, 0.7899]])  # Transition probability matrix (monthly)
P_daily = np.array([[0.9979, 0.0021], [0.0120, 0.9880]])  # Transition probability matrix (Daily )

p_monthly1=expm(P_daily)  # Compute the matrix exponential of P
p_daily1=np.pow(expm(p_monthly),(1/5))  # Compute the matrix exponential of P
#print(p_monthly1)
#print(p_daily1)    


# Output:

import numpy as np


# Given Monthly Transition Matrix (P1)
PM = np.array([
    [0.9629, 0.0371],
    [0.2102, 0.7899]
])

# Compute generator matrix Q
Q = logm(PM)

# Compute transition matrices for t=5 and t=20 months
PW = expm(Q / 5)
PD = expm(Q / 20)

# Display Results
print("Monthly Transition Matrix (P1):\n", PM)
print("\nGenerator Matrix (Q):\n", Q)
print("\nTransition Matrix for t=5 days (P5):\n", PW)
print("\nTransition Matrix for t=1 days (P20):\n", PD)
