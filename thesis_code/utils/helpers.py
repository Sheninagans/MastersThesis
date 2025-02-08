import pandas as pd
import numpy as np
import os

def load_data(filename):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        print(f"❌ Debugging: Python is looking for this file → {file_path}")
        return None

    return pd.read_csv(file_path)


def export_to_excel(data, filename="output.csv", index=False, sheet_name="Sheet1", file_format="csv"):
    """
    Export a pandas DataFrame or NumPy array to a CSV or Excel file in the 'results' directory.
    
    Parameters:
        data (pd.DataFrame or np.ndarray): Data to export.
        filename (str): Name of the output file.
        index (bool): Whether to include the index in the file.
        sheet_name (str): Name of the sheet in the Excel file (only for Excel format).
        file_format (str): Format of the output file, either 'csv' or 'excel'.
    
    Returns:
        str: Full path of the saved file.
    """
    try:
        results_path = os.path.join(os.getcwd(), "results")  # Define results directory
        os.makedirs(results_path, exist_ok=True)  # Ensure 'results' folder exists
        
        full_path = os.path.join(results_path, filename)  # Create full path
        
        # Convert NumPy array to DataFrame if necessary
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Ensure data is a DataFrame before exporting
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or a NumPy array.")
        
        # Export as CSV or Excel
        if file_format == "csv" or filename.endswith(".csv"):
            data.to_csv(full_path, index=index)
        elif file_format == "excel" or filename.endswith(".xlsx"):
            with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
                data.to_excel(writer, index=index, sheet_name=sheet_name)
        else:
            raise ValueError("Invalid file format. Choose 'csv' or 'excel'.")
        
        print(f"✅ Data successfully exported to: {full_path}")
        
        return full_path  # Return the file path for reference
    except Exception as e:
        print(f"❌ Error exporting data: {e}")




#Example Usage:
#export_to_excel(dataframe, filename="my_data.xlsx", sheet_name="Data", index=True)