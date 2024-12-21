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

    # Metrics storage
    model_names = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    train_times = []
    pred_times = []

    # Define models
    models = [
        # ("One-class SVM", unsupervised_models.train_one_class_svm),
        ("Isolation Forest", unsupervised_models.train_isolation_forest),
        ("Elliptic Envelope", unsupervised_models.train_elliptic_envelope),
        ("LOF", unsupervised_models.train_local_outlier_factor),
        ("HDBSCAN", unsupervised_models.train_hdbscan)
    ]

    for model_name, model_function in models:
        print(f"\nTraining and evaluating {model_name}...")

        # Train and predict
        if model_name == "HDBSCAN":
            model, y_pred, train_time, pred_time = model_function(
                X_train_pca, X_test_pca, min_samples=5, min_cluster_size=5, outlier_threshold=0.9)
        else:
            model, y_pred, train_time, pred_time = model_function(
                X_train_pca, X_test_pca)

        # Evaluate model
        precision, recall, f1, roc_auc = model_evaluation.evaluate_anomaly_detection(
            y_test, y_pred, model_name, train_time, pred_time)

        # Store metrics
        model_names.append(model_name)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        train_times.append(train_time)
        pred_times.append(pred_time)

        # Visualization: Anomaly Detection Results
        print(f"Visualizing results for {model_name}...")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                    c=y_pred, cmap='coolwarm', alpha=0.5)
        plt.title(f"Anomaly Detection Results: {model_name}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Prediction (-1: Anomaly, 1: Normal)")
        plt.show()

    # Summary Table
    print("\nSummary of Model Performances:")
    for i, name in enumerate(model_names):
        print(
            f"{name} -> Precision: {precisions[i]:.2f}, Recall: {recalls[i]:.2f}, "
            f"F1-Score: {f1_scores[i]:.2f}, ROC-AUC: {roc_aucs[i]:.2f}, "
            f"Train Time: {train_times[i]:.2f}s, Prediction Time: {pred_times[i]:.2f}s"
        )


if __name__ == "__main__":
    run_test()
