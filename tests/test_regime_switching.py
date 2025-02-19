import numpy as np
import pandas as pd 
import sys
import os

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from thesis_code.analysis.algo4 import fit
from thesis_code.utils.helpers import load_data, export_to_excel
from thesis_code.analysis.algo1 import generate_features

from sklearn.cluster import KMeans
from scipy.special import softmax

def initialize_S_randomized(Y, num_states, variability=0.2, temperature=0.8):
    """
    Initializes the state matrix S using k-means++ clustering with randomized soft assignments.

    Parameters:
    Y (numpy.ndarray or pd.DataFrame): The data matrix (T x D).
    num_states (int): The number of states/clusters.
    variability (float): Degree of randomness in cluster centers.
    temperature (float): Softmax temperature to control sharpness of assignments.

    Returns:
    S (numpy.ndarray): The initialized state matrix (T x num_states) with soft assignments.
    """

    # Ensure Y is a NumPy array
    Y_array = Y if isinstance(Y, np.ndarray) else Y.to_numpy()

    # Apply k-means++ clustering
    kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=10)
    kmeans.fit(Y_array)

    # Perturb cluster centers for variability
    cluster_centers = kmeans.cluster_centers_
    random_shifts = np.random.normal(scale=variability, size=cluster_centers.shape)
    perturbed_centers = cluster_centers + random_shifts  # Create randomized anchors

    # Compute distances to perturbed cluster centers
    distances = np.linalg.norm(Y_array[:, None, :] - perturbed_centers, axis=2)

    # Convert distances to probabilities using softmax
    S = softmax(-distances / temperature, axis=1)  # Higher temp = more uniform assignments

    return S



# Load Data
nas = load_data("nasdaq_1996_2005.csv")

# Ensure index is datetime
nas.index = pd.to_datetime(nas.index, format="%Y-%m-%d", errors='coerce')
nas = nas.sort_index()  # Ensure chronological order

# Compute log returns
nas["Log Returns"] = np.log(nas["Last Price"] / nas["Last Price"].shift(1))
nas["Scaled Prices"] = nas["Last Price"] / nas["Last Price"].iloc[-14]*100

# Drop NaNs before cumulative sum
nas = nas.dropna(subset=["Log Returns"])
export_to_excel(nas, filename="Log_Returns.csv")


# Generate features using the corrected column name
Y = generate_features(nas["Log Returns"], window_lengths=[6, 14])

Initial_centroids = []  # Empty list to store each S vector

# Generate 10 different S initializations and store them as columns
for i in range(10):
    Initial_centroids.append(initialize_S_randomized(Y, 2))  # Generate an S vector (T x 2)

# Convert list to NumPy array and stack as columns
Initial_centroids = np.column_stack(Initial_centroids)  # Final shape: (T, 10 * 2)

# Convert to DataFrame
Initial_centroids_df = pd.DataFrame(Initial_centroids)

# Export to Excel
export_to_excel(Initial_centroids_df, filename="Initial_centroids.csv")

print("✅ Initial centroids successfully exported!")
