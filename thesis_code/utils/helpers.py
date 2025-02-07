import pandas as pd
import os

def load_data(filename):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        print(f"❌ Debugging: Python is looking for this file → {file_path}")
        return None

    return pd.read_csv(file_path)

def export_to_excel(df, filename="output.xlsx", sheet_name="Sheet1", index=False):
    """
    Export a pandas DataFrame to an Excel file in the current directory.
    
    Returns:
        str: Full path of the saved Excel file.
    """
    try:
        current_path = os.getcwd()  # Get current working directory
        full_path = os.path.join(current_path, filename)  # Create full path
        
        df.to_excel(full_path, sheet_name=sheet_name, index=index, engine='openpyxl')
        print(f" Data successfully exported to: {full_path}")
        
        return full_path  # Return the file path for reference
    except Exception as e:
        print(f" Error exporting to Excel: {e}")
#Example Usage:
#export_to_excel(dataframe, filename="my_data.xlsx", sheet_name="Data", index=True)