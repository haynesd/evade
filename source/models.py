import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train_isolation_forest(X_train, X_test, use_percentile_threshold=True):
    """
    Train an Isolation Forest model and generate predictions and decision scores.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        use_percentile_threshold (bool): Whether to use percentile thresholding on decision scores.

    Returns:
        model: Trained Isolation Forest model.
        y_pred (np.ndarray): Binary predictions for test data (1 = anomaly).
        decision_scores (np.ndarray): Anomaly scores (higher = more normal).
    """
    model = IsolationForest(contamination=0.15, random_state=42)

    # Train and track memory usage
    tracemalloc.start()
    start_train = time.time()
    model.fit(X_train)
    train_time = (time.time() - start_train) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Training took {train_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    # Predict and track memory usage
    tracemalloc.start()
    start_pred = time.time()
    decision_scores = model.decision_function(X_test)
    if use_percentile_threshold:
        threshold = np.percentile(decision_scores, 15)
        y_pred = (decision_scores < threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = (time.time() - start_pred) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Prediction took {pred_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    return model, y_pred, decision_scores


def train_elliptic_envelope(X_train, X_test, use_percentile_threshold=True):
    """
    Train an Elliptic Envelope model and generate predictions and decision scores.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        use_percentile_threshold (bool): Whether to use percentile thresholding on decision scores.

    Returns:
        model: Trained Elliptic Envelope model.
        y_pred (np.ndarray): Binary predictions for test data (1 = anomaly).
        decision_scores (np.ndarray): Anomaly scores (higher = more normal).
    """
    model = EllipticEnvelope(
        contamination=0.15, support_fraction=0.8, random_state=42)

    # Train and track memory
    tracemalloc.start()
    start_train = time.time()
    model.fit(X_train)
    train_time = (time.time() - start_train) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Training took {train_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    # Predict and track memory
    tracemalloc.start()
    start_pred = time.time()
    decision_scores = model.decision_function(X_test)
    if use_percentile_threshold:
        threshold = np.percentile(decision_scores, 15)
        y_pred = (decision_scores < threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = (time.time() - start_pred) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Prediction took {pred_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    return model, y_pred, decision_scores


def train_lof(X_train, X_test, use_percentile_threshold=True):
    """
    Train a Local Outlier Factor model (with novelty detection) and predict test labels.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        use_percentile_threshold (bool): Whether to use percentile thresholding on decision scores.

    Returns:
        model: Trained LOF model.
        y_pred (np.ndarray): Binary predictions for test data (1 = anomaly).
        decision_scores (np.ndarray): Anomaly scores (higher = more normal).
    """
    model = LocalOutlierFactor(
        n_neighbors=50, contamination=0.15, novelty=True)

    # Train and track memory
    tracemalloc.start()
    start_train = time.time()
    model.fit(X_train)
    train_time = (time.time() - start_train) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Training took {train_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    # Predict and track memory
    tracemalloc.start()
    start_pred = time.time()
    decision_scores = model.decision_function(X_test)
    if use_percentile_threshold:
        threshold = np.percentile(decision_scores, 15)
        y_pred = (decision_scores < threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = (time.time() - start_pred) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Prediction took {pred_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    return model, y_pred, decision_scores


def train_one_class_svm(X_train, X_test, use_percentile_threshold=True):
    """
    Train a One-Class SVM model and generate anomaly predictions.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        use_percentile_threshold (bool): Whether to use percentile thresholding on decision scores.

    Returns:
        model: Trained One-Class SVM model.
        y_pred (np.ndarray): Binary predictions for test data (1 = anomaly).
        decision_scores (np.ndarray): Anomaly scores (higher = more normal).
    """
    model = OneClassSVM(kernel="rbf", gamma='scale', nu=0.01)

    # Train and track memory
    tracemalloc.start()
    start_train = time.time()
    model.fit(X_train)
    train_time = (time.time() - start_train) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Training took {train_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    # Predict and track memory
    tracemalloc.start()
    start_pred = time.time()
    decision_scores = model.decision_function(X_test)
    if use_percentile_threshold:
        threshold = np.percentile(decision_scores, 15)
        y_pred = (decision_scores < threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = (time.time() - start_pred) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Prediction took {pred_time:.2f} ms | Peak Memory: {peak / 1024:.2f} KB")

    return model, y_pred, decision_scores


def visualize_elliptic_envelope(X_train, X_test, model, decision_scores, n_components=2):
    """
    Visualize Elliptic Envelope decision boundary after reducing data to 2D with PCA.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        model: Trained Elliptic Envelope model.
        decision_scores (np.ndarray): Anomaly scores from the model.
        n_components (int): Number of PCA components to reduce to (default is 2).
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=n_components)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Re-train on 2D data for visualization
    vis_model = EllipticEnvelope(
        contamination=0.10, support_fraction=0.9, random_state=42)
    vis_model.fit(X_train_2d)

    # Create grid for contour
    xx, yy = np.meshgrid(
        np.linspace(X_train_2d[:, 0].min() - 1,
                    X_train_2d[:, 0].max() + 1, 500),
        np.linspace(X_train_2d[:, 1].min() - 1,
                    X_train_2d[:, 1].max() + 1, 500)
    )
    Z = vis_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=decision_scores,
                cmap='coolwarm', s=50, edgecolor='k', label='Test Data')
    plt.colorbar(label='Decision Score')
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                color='green', s=20, label='Training Data', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.title("Elliptic Envelope - Outlier Detection with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
