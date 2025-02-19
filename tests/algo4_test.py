import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from thesis_code.analysis.algo4 import fit
from thesis_code.utils.helpers import load_data, export_to_excel
from thesis_code.analysis.algo1 import generate_features

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
export_to_excel(Y, filename="Features_Stand.csv")

# Ensure 'Date' column exists and convert it to datetime
if 'Date' in nas.columns:
    nas['Date'] = pd.to_datetime(nas['Date'], format="%Y-%m-%d", errors='coerce')
    nas.set_index('Date', inplace=True)  # Set 'Date' as index

# Run the model
theta, S = fit(Y, num_states=2, lambda_penalty=1000, grid_size=0.05, tolerance=1e-9)
export_to_excel(theta, filename="theta.csv")

# Extract time index and prices
time_index = nas.index[:len(Y)]  # Select the most recent time indices
Scaled_prices = nas["Scaled Prices"].iloc[:len(Y)].values  # Select the most recent values
col_with_most_zeros = np.argmax(np.sum(S == 0, axis=0))  # Find the column with the most zeros
state_1_prob = S[:, col_with_most_zeros]  # Assign state_1_prob to that column


# Ensure no NaN values before plotting
if time_index.isna().any() or np.isnan(Scaled_prices).any():
    raise ValueError("time_index or scaled_prices contain NaN values. Check data preprocessing.")

# Plot state probability and log scale price
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Date")
ax1.set_ylabel("State 1 Probability")
ax1.plot(time_index, state_1_prob, color="tab:blue", label="State 1 Probability")
ax1.tick_params(axis="y")
ax1.set_xticks(time_index[::len(time_index) // 10])  # Show approximately 10 evenly spaced dates
ax1.set_xticklabels(time_index[::len(time_index) // 10].strftime('%Y-%m-%d'), rotation=45)

# Create second y-axis with log scale
ax2 = ax1.twinx()
ax2.set_ylabel("Indexed Prices")
ax2.plot(time_index, Scaled_prices, color="tab:orange", linestyle="dashed", label="Log Returns")
ax2.set_yscale("log")
ax2.tick_params(axis="y")

ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_formatter(ScalarFormatter())

fig.tight_layout()
plt.title("State 1 Probability and Scaled Log Index Price Over Time (Log Scale)")
plt.show()
