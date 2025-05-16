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


