from sklearn.model_selection import train_test_split
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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


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

    # Data cleaning
    data = data.drop_duplicates()
    X = data[selected_features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    data["Label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    y = data["Label"]

    # Feature engineering
    # Total Flags: Sum of all TCP flags (e.g., SYN, RST, ACK) to capture overall connection activity. High flag activity often indicates anomalies like SYN floods in DDoS attacks.
    # Protocol Aggregation: Combined metrics of protocol-based features (e.g., HTTP, DNS, IRC) to highlight patterns, as protocols like IRC and DNS are commonly exploited in botnet communications.
    # Packet Rate Normalization: Ratio of packet rate to total TCP flags, identifying disproportionate activity relative to flag usage, a common sign of anomalies.
    # Inter-Arrival Time (IAT) Ratio: Captures temporal patterns in traffic, with anomalies often showing irregular IAT values.
    # Rate-IAT Product: Product of packet rate and IAT, highlighting disrupted relationships between traffic volume and timing in anomalies.
    # Flag Ratios: Ratios of individual TCP flags to the total flag count (e.g., SYN or RST), emphasizing behaviors like SYN floods or RST storms indicative of anomalies.
    # Weighted Protocol Activity: Combination of protocol type and activity level, revealing interactions that often indicate misuse or malicious behavior.
    X['Total_Flags'] = X[['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                          'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                          'cwr_flag_number']].sum(axis=1)
    X['Protocol_Activity'] = X[[
        'HTTP', 'HTTPS', 'DNS', 'IRC', 'SSH']].sum(axis=1)
    X['Rate_Per_Flag'] = X['Rate'] / (X['Total_Flags'] + 1)
    X['IAT_Ratio'] = X['IAT'] / (X['Rate'] + 1)
    X['Rate_IAT'] = X['Rate'] * X['IAT']
    X['TotSize_Rate'] = X['Tot size'] / (X['Rate'] + 1)
    X['SYN_Ratio'] = X['syn_flag_number'] / (X['Total_Flags'] + 1)
    X['RST_Ratio'] = X['rst_flag_number'] / (X['Total_Flags'] + 1)
    X['Weighted_Protocol'] = X['Protocol Type'] * X['Protocol_Activity']

    # Train-test split
    X_benign = X[y == 0]
    X_anomalies = X[y == 1]
    train_benign_size = int(len(X_benign) * 0.9)
    X_train = X_benign.sample(n=train_benign_size, random_state=42)
    y_train = pd.Series(0, index=X_train.index)
    test_benign_size = int(len(X_benign) * 0.1)
    test_anomalies_size = int(test_benign_size * 0.2)
    X_test_benign = X_benign.drop(X_train.index).sample(
        n=test_benign_size, random_state=42)
    X_test_anomalies = X_anomalies.sample(
        n=test_anomalies_size, random_state=42)
    X_test = pd.concat([X_test_benign, X_test_anomalies], axis=0)
    y_test = pd.concat([pd.Series(0, index=X_test_benign.index),
                        pd.Series(1, index=X_test_anomalies.index)], axis=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Display packet counts
    print(f"Number of benign packets in training: {len(X_train)}")
    print(f"Number of benign packets in testing: {len(X_test_benign)}")
    print(f"Number of anomaly packets in testing: {len(X_test_anomalies)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA for dimensionality reduction
    pca = PCA()
    pca.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Selected PCA components: {n_components}")
    pca = PCA(n_components=n_components)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train_scaled))
    X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))

    return X_train_pca, y_train, X_test_pca, y_test
