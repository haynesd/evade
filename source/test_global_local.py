import data_pca
import data_processing
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import hdbscan
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the performance of a model using accuracy, precision, recall, and F1-score.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels (-1 for anomalies, 1 for normal points).
        model_name (str): Name of the model being evaluated.

    Returns:
        None
    """
    # Convert labels to 0 (anomalies) and 1 (normal) for compatibility
    y_true_binary = [0 if y == -1 else 1 for y in y_true]
    y_pred_binary = [0 if y == -1 else 1 for y in y_pred]

    # Compute metrics
    acc = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, pos_label=1)
    recall = recall_score(y_true_binary, y_pred_binary, pos_label=1)
    f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"  Accuracy: {acc:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(
        f"  Confusion Matrix:\n{confusion_matrix(y_true_binary, y_pred_binary)}\n")


def run_test():
    base_dir = os.getcwd()  # Current working directory
    # Path to /nad/data folder
    data_folder = os.path.join(base_dir, "data")
    # Path to test.csv file
    csv_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")

    # Load and preprocess the data
    X_train, y_train, X_test, y_test, feature_names = data_processing.load_and_preprocess_data(
        csv_file
    )

    # Perform PCA and get transformed datasets
    print("Performing PCA...")
    pca = PCA(n_components=10)  # Adjust the number of components as needed
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA completed. X_train_pca shape: {X_train_pca.shape}")
    print(f"X_test_pca shape: {X_test_pca.shape}")

    # Convert y_test to match Elliptic Envelope and HDBSCAN label conventions (-1 for anomalies, 1 for normal)
    y_test_converted = [1 if y == 1 else -1 for y in y_test]

    # Global Anomalies Detection with Elliptic Envelope
    print("Training Isolation Forest for global anomalies...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(X_train_pca)

    # Predict global anomalies
    global_anomalies = isolation_forest.predict(
        X_test_pca)  # -1 for anomalies, 1 for inliers
    print(f"Global anomalies detected: {sum(global_anomalies == -1)}")
    evaluate_model(y_test_converted, global_anomalies,
                   "Elliptic Envelope (Global)")

    # Local Anomalies Detection with HDBSCAN
    print("Training HDBSCAN for local anomalies...")
    hdbscan_model = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10)
    hdbscan_labels = hdbscan_model.fit_predict(
        X_test_pca)  # -1 for noise (local anomalies)

    # Convert HDBSCAN labels to match Elliptic Envelope (-1 for anomalies, 1 for normal)
    hdbscan_preds = [1 if label != -1 else -1 for label in hdbscan_labels]
    print(f"Local anomalies detected: {sum(hdbscan_labels == -1)}")
    evaluate_model(y_test_converted, hdbscan_preds, "HDBSCAN (Local)")

    # Plot global and local anomalies
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                c='blue', label='Normal Points', alpha=0.5)
    plt.scatter(
        X_test_pca[global_anomalies == -1, 0],
        X_test_pca[global_anomalies == -1, 1],
        c='red',
        label='Global Anomalies (Isolation Forest)',
        alpha=0.7,
    )
    plt.scatter(
        X_test_pca[hdbscan_labels == -1, 0],
        X_test_pca[hdbscan_labels == -1, 1],
        c='orange',
        label='Local Anomalies (HDBSCAN)',
        alpha=0.7,
    )
    plt.title("Global and Local Anomalies")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_test()
