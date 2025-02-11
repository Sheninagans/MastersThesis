import numpy as np
from thesis_code.analysis.algo4 import fit_state_sequence  # Import your function

# Step 1: Create a simple observation matrix Y (T x D)
Y = np.array([[1, 0.2],   # Observation at t=0
              [0.5, 0.2],   # Observation at t=1
              [1.2, 2.1],
              [9, 8],
              [5.5, 10],])  # Observation at t=2

# Step 2: Define known state parameters theta (num_states x D)
theta = np.array([[1, 1],   # State 0
                  [2, 2]])  # State 1

# Step 3: Create a small probability simplex grid C (num_states x N)
C = np.array([[0.8, 0.2],   # First candidate probability vector
              [0.2, 0.8]])  # Second candidate probability vector

# Step 4: Define a small lambda penalty
lambda_penalty = 1

# Step 5: Run fit_state_sequence with controlled values

S_test = fit_state_sequence(Y, theta, C, lambda_penalty, num_states=2)

# Step 6: Print the estimated S
print("Estimated S:")
print(S_test)

print("Row sums of S_test:", S_test.sum(axis=1))

for row in S_test:
    assert any(np.allclose(row, C[:, i]) for i in range(C.shape[1])), "S contains invalid values!"
print("✅ S contains only valid probability simplex vectors from C!")
