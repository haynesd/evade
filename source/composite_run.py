import data_pca
import unsupervised_models
import data_processing
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
import warnings
import model_evaluation

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_test():
    base_dir = os.getcwd()  # Current working directory
    data_folder = os.path.join(base_dir, "data")
    csv_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")

    # Load and preprocess data
    X_train, y_train, X_test, y_test, feature_names = data_processing.load_and_preprocess_data(
        csv_file)

    print(f"Unique values in y_test (ground truth): {set(y_test)}")

    # Perform PCA to reduce dimensionality
    print("Performing PCA...")

    X_train_pca, X_test_pca = data_pca.perform_pca_and_plot(
        X_train, y_train, feature_names, X_test=X_test)

    print(
        f"X_train_pca shape: {X_train_pca.shape}, X_test_pca shape: {X_test_pca.shape}")

    # Run Isolation Forest first
    print("Running Isolation Forest...")
    isolation_forest, iso_y_pred, iso_train_time, iso_pred_time = unsupervised_models.train_isolation_forest(
        X_train_pca, X_test_pca)

    # Evaluate Isolation Forest results
    iso_precision, iso_recall, iso_f1, iso_roc_auc = model_evaluation.evaluate_anomaly_detection(
        y_test, iso_y_pred, "Isolation Forest", iso_train_time, iso_pred_time)

    # Visualize Isolation Forest Results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                c=iso_y_pred, cmap='coolwarm', alpha=0.5)
    plt.title("Anomaly Detection Results: Isolation Forest")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Prediction (-1: Anomaly, 1: Normal)")
    plt.show()

    # Filter anomalies detected by Isolation Forest (retain only anomalies)
    X_test_anomalies = X_test_pca[iso_y_pred == -1]
    print(
        f"Filtered X_test_pca shape with anomalies: {X_test_anomalies.shape}")

    # Run HDBSCAN on the anomalies detected by Isolation Forest
    print("Running HDBSCAN on Isolation Forest anomalies...")
    # Dynamically adjust HDBSCAN parameters based on the anomaly set size
    min_cluster_size = max(5, len(X_test_anomalies) // 100)
    min_samples = max(5, len(X_test_anomalies) // 100)

    hdbscan_model, hdbscan_y_pred, hdbscan_train_time, hdbscan_pred_time = unsupervised_models.train_hdbscan(
        X_test_anomalies,
        X_test_anomalies,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        outlier_threshold=0.7  # Lower the threshold for stricter clustering
    )

    # Visualize HDBSCAN Results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_anomalies[:, 0], X_test_anomalies[:, 1],
                c=hdbscan_y_pred, cmap='viridis', alpha=0.5)
    plt.title("Clustering Results: HDBSCAN on Isolation Forest Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster ID")
    plt.show()

    # Evaluate HDBSCAN results (use ground truth for anomalies if applicable)
    hdbscan_precision, hdbscan_recall, hdbscan_f1, hdbscan_roc_auc = model_evaluation.evaluate_anomaly_detection(
        y_test[iso_y_pred == -1], hdbscan_y_pred, "HDBSCAN", hdbscan_train_time, hdbscan_pred_time)

    # Summary Table
    print("\nSummary of Model Performances:")
    print(
        f"Isolation Forest -> Precision: {iso_precision:.2f}, Recall: {iso_recall:.2f}, "
        f"F1-Score: {iso_f1:.2f}, ROC-AUC: {iso_roc_auc:.2f}, "
        f"Train Time: {iso_train_time:.2f}s, Prediction Time: {iso_pred_time:.2f}s"
    )
    print(
        f"HDBSCAN on Isolation Forest Anomalies -> Precision: {hdbscan_precision:.2f}, Recall: {hdbscan_recall:.2f}, "
        f"F1-Score: {hdbscan_f1:.2f}, ROC-AUC: {hdbscan_roc_auc:.2f}, "
        f"Train Time: {hdbscan_train_time:.2f}s, Prediction Time: {hdbscan_pred_time:.2f}s"
    )


if __name__ == "__main__":
    run_test()
