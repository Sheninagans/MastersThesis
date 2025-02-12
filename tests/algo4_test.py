import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, FuncFormatter
import pandas as pd
# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from thesis_code.analysis.algo4 import fit
from thesis_code.utils.helpers import load_data
from thesis_code.analysis.algo1 import generate_features
from thesis_code.utils.helpers import export_to_excel



# Load data
nas = load_data("nasdaq_1996_2005.csv")

# Ensure index is datetime
nas.index = pd.to_datetime(nas.index, format="%Y-%m-%d", errors='coerce')  # Convert index properly
nas = nas.sort_index()  # Ensure chronological order


# Scale data to start at index 100
nas["Scaled Price"] = (nas["Last Price"] / nas["Last Price"].iloc[-1]) * 100
nas = nas.dropna(subset=["Scaled Price"])  # Remove rows with missing prices
export_to_excel(nas,filename="Nas1.csv")
# Generate features
Y = generate_features(nas["Scaled Price"], window_lengths=[6, 14])
Y = Y.iloc[13:]

# Ensure 'Date' column exists and convert it to datetime
if 'Date' in nas.columns:
    nas['Date'] = pd.to_datetime(nas['Date'], format="%Y-%m-%d", errors='coerce')
    nas.set_index('Date', inplace=True)  # Set 'Date' as index

# Run the model
theta, S = fit(Y, num_states=2, lambda_penalty=100, grid_size=0.05)
export_to_excel(theta, filename="theta.csv")

# Extract time index and prices
time_index = nas.index[-len(Y):]  # Align time index with Y
scaled_prices = nas["Scaled Price"].iloc[-len(Y):]  # Scaled index price
state_1_prob = S[:, 0]  # Probability of being in state 1

print("NaN check - Time Index:", nas.index.isna().sum())

# Ensure no NaN values before plotting
if time_index.isna().any() or np.isnan(scaled_prices).any():
    raise ValueError("time_index or scaled_prices contain NaN values. Check data preprocessing.")

# Plot state probability and log scale price
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Date")
ax1.set_ylabel("State 1 Probability")
ax1.plot(time_index, state_1_prob, color="tab:blue", label="State 1 Probability")
ax1.tick_params(axis="y")
ax1.set_xticks(time_index[::len(time_index)//10])  # Show approximately 10 evenly spaced dates
ax1.set_xticklabels(time_index[::len(time_index)//10].strftime('%Y-%m-%d'), rotation=45)

# Create second y-axis with log scale
ax2 = ax1.twinx()
ax2.set_ylabel("Scaled Index Price")
ax2.set_yscale("log")  # Apply log scale to the axis
ax2.plot(time_index, scaled_prices, color="tab:orange", linestyle="dashed", label="Scaled Index Price")
ax2.tick_params(axis="y")
#ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
#ax2.yaxis.get_offset_text().set_visible(False)  # Format y-axis as plain numbers
ax2.yaxis.set_major_formatter(FuncFormatter(mticker.ScalarFormatter()))
ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
#ax2.yaxis.set_major_formatter(ScalarFormatter())
#ax2.yaxis.get_major_formatter().set_scientific(False)

fig.tight_layout()
plt.title("State 1 Probability and Scaled Index Price Over Time (Log Scale)")
plt.show()
