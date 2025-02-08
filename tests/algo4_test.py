import numpy as np
from thesis_code.analysis.algo4 import fit
from thesis_code.utils.helpers import load_data
from thesis_code.analysis.algo1 import generate_features
from thesis_code.utils.helpers import export_to_excel


nas = load_data("nasdaq_2013_2022.csv")
# Create a simple dataset
Y = generate_features(nas["Last Price"], window_lengths=[6, 14])

#print(nas['Last Price'])
#print(Y.head(5))
#export_to_excel(Y)
Y=Y.iloc[13:]

# Run the model
theta, S = fit(Y, num_states=2, lambda_penalty=1000)
export_to_excel(S, filename="S.csv")
export_to_excel(theta,filename="theta.csv")
print(theta[:10])
print(S[:10])

""" # Debug prints
print("Final theta (cluster means):\n", theta)
print("Final state assignments:\n", S)

# Assertions for debugging
assert theta is not None, "Theta was not computed!"
assert S is not None, "State sequence was not computed!"
assert S.shape == (Y.shape[0], 2), "State sequence dimensions are incorrect!" """
