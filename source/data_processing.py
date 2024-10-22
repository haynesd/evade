import data_preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_file):
    """
    Load the dataset from CSV and preprocess the features and labels.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        tuple: Processed train/test features and labels.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Prepare data
    X, y = data_preparation.prepare_data(data) # type: ignore

    # Check consistency of X and y
    print(f"X shape: {X.shape}, y shape: {y.shape}")
   
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    # Check if the splitting is consistent
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test
