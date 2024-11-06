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


def prepare_data(df, selected_features=None):
    """
    Prepare the features (X) and labels (y) from the DataFrame based on selected features, maintaining a specific order.

    Args:
        df (pd.DataFrame): The loaded DataFrame.
        selected_features (list of str): List of feature names to include, in order of preference.

    Returns:
        tuple: Features (X), labels (y), and list of feature names after processing.
    """
    # Feature options in the specified order with existence checks
    feature_options = {
        'sport': lambda df: df['sport'] if 'sport' in df.columns else pd.Series(),
        'dsport': lambda df: df['dsport'] if 'dsport' in df.columns else pd.Series(),
        'sttl': lambda df: df['sttl'] if 'sttl' in df.columns else pd.Series(),
        'total_len': lambda df: df['total_len'] if 'total_len' in df.columns else pd.Series(),
        'stime': lambda df: df['stime'] if 'stime' in df.columns else pd.Series(),
        'srcip_numeric': lambda df: df['srcip'].apply(ip_to_numeric) if 'srcip' in df.columns else pd.Series(),
        'dstip_numeric': lambda df: df['dstip'].apply(ip_to_numeric) if 'dstip' in df.columns else pd.Series(),
        'protocol_m_tcp': lambda df: pd.get_dummies(df, columns=['protocol_m'], drop_first=True).filter(regex='_tcp$') if 'protocol_m' in df.columns else pd.DataFrame(),
        'protocol_m_udp': lambda df: pd.get_dummies(df, columns=['protocol_m'], drop_first=True).filter(regex='_udp$') if 'protocol_m' in df.columns else pd.DataFrame(),
        'payload_size': lambda df: df['payload'].apply(lambda x: len(str(x))) if 'payload' in df.columns else pd.Series(),
        'payload_size_ratio': lambda df: df.apply(lambda row: row['payload_size'] / row['total_len'] if row['total_len'] > 0 else 0, axis=1) if 'payload_size' in df.columns and 'total_len' in df.columns else pd.Series(),
        'inter_arrival_time': lambda df: df['stime'].diff().fillna(0) if 'stime' in df.columns else pd.Series()
    }

    # Default to all features if none are specified
    if selected_features is None:
        selected_features = list(feature_options.keys())

    # Ensure the selected features follow the specified order
    selected_features = [
        feature for feature in feature_options.keys() if feature in selected_features]

    # Process selected features in the specified order
    processed_features = []
    for feature in selected_features:
        if feature in feature_options:
            df[feature] = feature_options[feature](df)
            processed_features.append(feature)

    # Drop original IP columns if they were converted
    if 'srcip_numeric' in processed_features:
        df.drop(columns=['srcip'], inplace=True)
    if 'dstip_numeric' in processed_features:
        df.drop(columns=['dstip'], inplace=True)

    # Drop payload column after deriving features
    if 'payload_size' in processed_features or 'payload_size_ratio' in processed_features:
        if 'payload' in df.columns:
            df.drop(columns=['payload'], inplace=True)

    # Prepare features (X) and labels (y) in the specified order
    X = df[processed_features]
    y = df['label'].apply(
        lambda x: 0 if x == 'Benign' else 1) if 'label' in df.columns else pd.Series()

    # Capture final feature names
    feature_names = X.columns.tolist()

    return X, y, feature_names
