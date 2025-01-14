import ipaddress
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ip_to_numeric(ip_string):
    """Convert IP addresses to numeric values."""
    try:
        return int(ipaddress.ip_address(ip_string))
    except ValueError:
        return 0


def prepare_data(df, selected_features=None):
    """Prepare dataset for model training."""
    feature_options = {
        'sport': lambda df: df['sport'] if 'sport' in df.columns else pd.Series(dtype=float),
        'dsport': lambda df: df['dsport'] if 'dsport' in df.columns else pd.Series(dtype=float),
        'sttl': lambda df: df['sttl'] if 'sttl' in df.columns else pd.Series(dtype=float),
        'total_len': lambda df: df['total_len'] if 'total_len' in df.columns else pd.Series(dtype=float),
        'stime': lambda df: df['stime'] if 'stime' in df.columns else pd.Series(dtype=float),
        'srcip_numeric': lambda df: df['srcip'].apply(ip_to_numeric) if 'srcip' in df.columns else pd.Series(dtype=float),
        'dstip_numeric': lambda df: df['dstip'].apply(ip_to_numeric) if 'dstip' in df.columns else pd.Series(dtype=float),
        'protocol_m_tcp': lambda df: pd.get_dummies(df, columns=['protocol_m'], drop_first=True).filter(regex='_tcp$') if 'protocol_m' in df.columns else pd.DataFrame(),
        'protocol_m_udp': lambda df: pd.get_dummies(df, columns=['protocol_m'], drop_first=True).filter(regex='_udp$') if 'protocol_m' in df.columns else pd.DataFrame(),
        'payload_size': lambda df: df['payload'].apply(lambda x: len(str(x))) if 'payload' in df.columns else pd.Series(dtype=float),
        'payload_size_ratio': lambda df: df.apply(lambda row: row['payload_size'] / row['total_len'] if row['total_len'] > 0 else 0, axis=1) if 'payload_size' in df.columns and 'total_len' in df.columns else pd.Series(dtype=float),
        'inter_arrival_time': lambda df: df['stime'].diff().fillna(0) if 'stime' in df.columns else pd.Series(dtype=float)
    }

    # Default to all features if none are specified
    if selected_features is None:
        selected_features = list(feature_options.keys())

    # Process selected features in the specified order
    processed_features = []
    for feature in selected_features:
        if feature in feature_options:
            df[feature] = feature_options[feature](df)
            processed_features.append(feature)

    # Drop original IP columns if converted
    if 'srcip_numeric' in processed_features:
        df.drop(columns=['srcip'], inplace=True)
    if 'dstip_numeric' in processed_features:
        df.drop(columns=['dstip'], inplace=True)

    # Drop payload column after deriving features
    if 'payload_size' in processed_features or 'payload_size_ratio' in processed_features:
        if 'payload' in df.columns:
            df.drop(columns=['payload'], inplace=True)

    # Prepare features (X) and labels (y)
    X = df[processed_features]
    y = df['label'].apply(
        lambda x: 0 if x == 'Benign' else 1) if 'label' in df.columns else pd.Series(dtype=int)

    return X, y


def getTrainTestDataFromCSV(csv_file, apply_pca=True):
    """Split dataset into training and test sets with specified anomaly ratio."""
    data = pd.read_csv(csv_file)

    # Select features
    selected_features = [
        'sport', 'dsport', 'sttl', 'total_len', 'stime', 'srcip_numeric',
        'dstip_numeric', 'protocol_m_tcp', 'protocol_m_udp', 'payload_size',
        'payload_size_ratio', 'inter_arrival_time'
    ]
    X, y = prepare_data(data, selected_features)

    # Separate benign and anomalies
    benign_data = X[y == 0]
    benign_labels = y[y == 0]
    anomaly_data = X[y == 1]
    anomaly_labels = y[y == 1]

    # Training set: 90% benign data
    train_benign_size = int(len(benign_data) * 0.9)
    X_train = benign_data.sample(n=train_benign_size, random_state=42)
    y_train = pd.Series(0, index=X_train.index)

    # Test set: 10% benign + anomalies
    test_benign_size = len(benign_data) - train_benign_size
    test_anomalies_size = min(len(anomaly_data), int(test_benign_size * 0.1))
    X_test_benign = benign_data.drop(X_train.index)
    X_test_anomalies = anomaly_data.sample(
        n=test_anomalies_size, random_state=42)

    # Combine test data
    X_test = pd.concat([X_test_benign, X_test_anomalies])
    y_test = pd.concat([pd.Series(0, index=X_test_benign.index),
                        pd.Series(1, index=X_test_anomalies.index)])
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Display packet counts
    print(f"Number of benign packets in training: {len(X_train)}")
    print(f"Number of benign packets in testing: {len(X_test_benign)}")
    print(f"Number of anomaly packets in testing: {len(X_test_anomalies)}")

    # Apply PCA
    if apply_pca:
        pca = PCA(n_components=min(X_train.shape[1], 10))
        X_train = pd.DataFrame(pca.fit_transform(X_train))
        X_test = pd.DataFrame(pca.transform(X_test))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test
