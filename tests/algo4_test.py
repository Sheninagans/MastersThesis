import numpy as np
from thesis_code.analysis.algo4 import fit
from thesis_code.utils.helpers import load_data
from thesis_code.analysis.algo1 import generate_features

nas = load_data("nasdaq_1996_2205.csv")
# Create a simple dataset
Y = generate_features(nas["Last Price"], window_lengths=[6, 14])
# Run the model
theta, S = fit(Y, num_states=2, lambda_penalty=1000)

# Debug prints
print("Final theta (cluster means):\n", theta)
print("Final state assignments:\n", S)

# Assertions for debugging
assert theta is not None, "Theta was not computed!"
assert S is not None, "State sequence was not computed!"
assert S.shape == (Y.shape[0], 2), "State sequence dimensions are incorrect!"
