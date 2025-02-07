from thesis_code.utils.helpers import load_data
import pandas as pd

nas = load_data("nasdaq_composite.csv")

# Convert 'Date' column to datetime
nas["Date"] = pd.to_datetime(nas["Date"], format="%d/%m/%Y")

# Split the data
data_1 = nas[(nas["Date"] >= "1996-01-01") & (nas["Date"] <= "2005-12-31")]
data_2 = nas[(nas["Date"] >= "2013-01-01") & (nas["Date"] <= "2022-12-31")]

# Print results
data_1.to_csv("data/nasdaq_1996_2005.csv", index=False)
data_2.to_csv("data/nasdaq_2013_2022.csv", index=False)