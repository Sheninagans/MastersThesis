import numpy as np
import pandas as pd

from thesis_code.utils.helpers import export_to_excel

###############   Algorithm 1     ########################
def generate_features(time_series, window_lengths=[6, 14]):
    """
    Generate features for the jump model.
    
    Parameters:
        time_series (list or np.array): The input is a univariate time series.
        window_lengths (list): List of window lengths for feature computation.
        
    Returns:
        pd.DataFrame: A DataFrame c ontaining the computed features.
    """
    time_series = np.array(time_series)
    T = len(time_series)
    
    # Initialize a dictionary to store features
    features = {
        "Observation": time_series,
        "Absolute Change": np.abs(np.diff(time_series, prepend=np.nan)),
        "Previous Absolute Change": np.abs(np.diff(time_series, n=1, prepend=[np.nan, np.nan])[:T]),
    }
    
    for w in window_lengths:
        # Compute rolling statistics
        features[f"Centered Mean (w={w})"] = pd.Series(time_series).rolling(window=w, center=False).mean()
        features[f"Centered Std (w={w})"] = pd.Series(time_series).rolling(window=w, center=False).std()
        features[f"Left Mean (w={w})"] = pd.Series(time_series).rolling(window=w).apply(lambda x: np.mean(x[:w//2]), raw=True)
        features[f"Left Std (w={w})"] = pd.Series(time_series).rolling(window=w).apply(lambda x: np.std(x[:w//2],ddof=1), raw=True)
        features[f"Right Mean (w={w})"] = pd.Series(time_series).rolling(window=w).apply(lambda x: np.mean(x[w//2:]), raw=True)
        features[f"Right Std (w={w})"] =pd.Series(time_series).rolling(window=w).apply(lambda x: np.std(x[w//2:],ddof=1), raw=True)
    return pd.DataFrame(features)
