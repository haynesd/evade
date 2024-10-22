import pandas as pd
import ipaddress

def ip_to_numeric(ip_string):
    """
    Convert IP address to a numeric format.
    
    Args:
        ip_string (str): IP address as string.
        
    Returns:
        int: Numeric representation of IP address.
    """
    try:
        return int(ipaddress.ip_address(ip_string))
    except ValueError:
        return 0  # Handle invalid IPs, or use NaN

def prepare_data(df):
    """
    Prepare the features (X) and labels (y) from the DataFrame.
    
    Args:
        df (pd.DataFrame): The loaded DataFrame.
        
    Returns:
        tuple: Features (X) and labels (y) after processing.
    """
    # Check the initial shape of the data
    print(f"Original data shape: {df.shape}")
    
    # Convert IP addresses to numeric values
    df['srcip_numeric'] = df['srcip'].apply(ip_to_numeric)
    df['dstip_numeric'] = df['dstip'].apply(ip_to_numeric)
    print(f"After IP conversion: {df.shape}")
    
    # Drop original IP columns
    df.drop(columns=['srcip', 'dstip'], inplace=True)
    print(f"After dropping IP columns: {df.shape}")
    
    # One-hot encode categorical columns (e.g., protocol_m)
    df = pd.get_dummies(df, columns=['protocol_m'], drop_first=True)
    print(f"After one-hot encoding: {df.shape}")
    
    # Add feature: Payload size (in bytes)
    df['payload_size'] = df['payload'].apply(lambda x: len(str(x)))  # Use size of payload
    print(f"After adding payload size: {df.shape}")
    
    # Add feature: Packet size ratio (payload size / total packet length)
    df['payload_size_ratio'] = df.apply(lambda row: row['payload_size'] / row['total_len'] if row['total_len'] > 0 else 0, axis=1)
    print(f"After adding payload size ratio: {df.shape}")
    
    # Add feature: Inter-packet arrival time
    df['inter_arrival_time'] = df['stime'].diff().fillna(0)  # Calculate time differences between packets
    print(f"After adding inter-packet arrival time: {df.shape}")
    
    # Drop payload content itself but keep the derived features
    df.drop(columns=['payload'], inplace=True)
    print(f"After dropping payload: {df.shape}")
    
    # Check for missing values (optional, for debugging purposes)
    print(f"Missing values in data: {df.isnull().sum()}")
    
    # Prepare features (X) and labels (y)
    X = df.drop(columns=['label'])
    y = df['label'].apply(lambda x: 0 if x == 'Benign' else 1)  # Binary classification
    
    # Check final X and y shapes before returning
    print(f"Final X shape: {X.shape}, Final y shape: {y.shape}")

    return X, y
