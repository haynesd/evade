import data_pca
import supervised_models
import unsupervised_models
import data_processing
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from time import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_model(y_true, y_pred, model_name, train_time, pred_time):
    # Convert labels to binary format if necessary
    y_true_binary = [0 if y == -1 else 1 for y in y_true]
    y_pred_binary = [0 if y == -1 else 1 for y in y_pred]

    acc = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1)
    recall = recall_score(y_true_binary, y_pred_binary, pos_label=1)
    precision = precision_score(y_true_binary, y_pred_binary, pos_label=1)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"  Accuracy: {acc:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Training Time: {train_time:.2f} seconds")
    print(f"  Prediction Time: {pred_time:.2f} seconds")
    print(f"  Confusion Matrix:\n{cm}\n")

    return acc, precision, recall, f1


def run_test():
    base_dir = os.getcwd()  # Current working directory
    data_folder = os.path.join(base_dir, "data")
    csv_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")

    # Load and preprocess data
    X_train, y_train, X_test, y_test, feature_names = data_processing.load_and_preprocess_data(csv_file)

    # Perform PCA
    print("Performing PCA...")
    X_train_pca, X_test_pca = data_pca.perform_pca_and_plot(
        X_train, y_train, feature_names, X_test=X_test
    )
    print(f"X_train_pca shape: {X_train_pca.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_pca shape: {X_test_pca.shape}, y_test shape: {y_test.shape}")

    # Metrics storage
    model_names = []
    accuracies = []
    precisions = []
    f1_scores = []
    recalls = []
    train_times = []
    pred_times = []

    # Define models
    models = [#("One-class SVM", unsupervised_models.train_one_class_svm, False)
        ("Random Forest", supervised_models.train_random_forest, True),
        ("Isolation Forest", unsupervised_models.train_isolation_forest, False),
        ("Elliptic Envelope", unsupervised_models.train_elliptic_envelope, False),
        ("LOF", unsupervised_models.train_local_outlier_factor, False),
        ("HDBSCAN", unsupervised_models.train_hdbscan, False)
    ]

    for model_name, model_function, is_supervised in models:
        print(f"Training and evaluating {model_name}...")

        if model_name == "Elliptic Envelope":
            model, y_pred, train_time, pred_time = model_function(X_train_pca, X_test_pca)
        elif model_name == "HDBSCAN":
            # HDBSCAN has specific parameters
            model, y_pred, train_time, pred_time = model_function(
                X_train_pca, X_test_pca, min_samples=5, min_cluster_size=10
            )
        elif is_supervised:
            # Supervised models
            model, acc, train_time, pred_time = model_function(
                X_train_pca, y_train, X_test_pca, y_test
            )
            y_pred = model.predict(X_test_pca)
        else:
            # Unsupervised models
            model, y_pred, train_time, pred_time = model_function(X_train_pca, X_test_pca)

        # Ensure y_pred is array-like
        if not isinstance(y_pred, (list, tuple, np.ndarray)):
            y_pred = np.array(y_pred)

        acc, precision, recall, f1 = evaluate_model(y_test, y_pred, model_name, train_time, pred_time)

        # Store results
        model_names.append(model_name)
        accuracies.append(acc)
        precisions.append(precision)
        f1_scores.append(f1)
        recalls.append(recall)
        train_times.append(train_time)
        pred_times.append(pred_time)

    # Visualization
    print("\nSummary of Model Performances:")
    for i, name in enumerate(model_names):
        print(f"{name} -> Accuracy: {accuracies[i]:.2f}, Precision: {precisions[i]:.2f}, F1-Score: {f1_scores[i]:.2f}, Recall: {recalls[i]:.2f}")
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies, label="Accuracy")
    plt.bar(model_names, f1_scores, label="F1-Score", alpha=0.7)
    plt.bar(model_names, precisions, label="Precision", alpha=0.5)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    run_test()
