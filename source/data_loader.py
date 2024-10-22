# data_loader.py
import pandas as pd

def load_labeled_data(csv_file):
    """
    Loads labeled data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the features and labels.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert label: 'Benign' to 0 and any other value to 1 (for malicious traffic)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    return df