from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Canadian Institute for Cybersecurity - CIC IoT Dataset 2023
#
# A real-time dataset and benchmark for large-scale attacks in IoT environment
# The main goal of this research is to propose a novel and extensive IoT attack
# dataset to foster the development of security analytics applications in real
# IoT operations. To accomplish this, 33 attacks are executed in an IoT topology
# composed of 105 devices.
#
# These attacks are classified into seven categories, namely DDoS, DoS, Recon,
# Web-based, Brute Force, Spoofing, and Mirai. Finally, all attacks are executed
# by malicious IoT devices targeting other IoT device
#
# https://www.unb.ca/cic/datasets/iotdataset-2023.html


def getTrainTestDataFromCSV(csv_file):
    data = pd.read_csv(csv_file)

    # Feature selection
    selected_features = [
        "Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number",
        "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
        "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count",
        "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum",
        "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance"
    ]

    # Remove duplicates
    data = data.drop_duplicates()

    # Filter features
    X = data[selected_features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    # Align labels
    data = data.loc[X.index]
    data["Label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Update X and y
    X = data[selected_features]
    y = data["Label"]

    # Reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Split benign and anomaly data
    X_benign = X[y == 0]
    X_anomalies = X[y == 1]

    train_benign_size = int(len(X_benign) * 0.9)
    X_train = X_benign.sample(n=train_benign_size, random_state=42)
    y_train = pd.Series(0, index=X_train.index)

    test_benign_size = int(len(X_benign) * 0.1)
    test_anomalies_size = min(len(X_anomalies), int(test_benign_size * 0.1))
    X_test_benign = X_benign.drop(X_train.index).sample(
        n=test_benign_size, random_state=42)
    X_test_anomalies = X_anomalies.sample(
        n=test_anomalies_size, random_state=42)

    # Combine and shuffle test data
    X_test = pd.concat([X_test_benign, X_test_anomalies], axis=0)
    y_test = pd.concat([pd.Series(0, index=X_test_benign.index),
                        pd.Series(1, index=X_test_anomalies.index)], axis=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Display packet counts
    print(f"Number of benign packets in training: {len(X_train)}")
    print(f"Number of benign packets in testing: {len(X_test_benign)}")
    print(f"Number of anomaly packets in testing: {len(X_test_anomalies)}")

    # Apply PCA
    pca = PCA(n_components=min(len(X_train.columns), 10))
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    X_test_pca = pd.DataFrame(pca.transform(X_test))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    return X_train_scaled, y_train, X_test_scaled, y_test
