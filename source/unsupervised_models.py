import numpy as np
from time import time
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import hdbscan


def train_one_class_svm(X_train, X_test):
    """
    Train One-Class Support Vector Machine (SVM) and predict anomalies on the test set.
    """
    start_train = time()
    one_class_svm = OneClassSVM(gamma='auto')
    one_class_svm.fit(X_train)
    train_time = time() - start_train

    start_pred = time()
    y_pred = one_class_svm.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time() - start_pred

    return one_class_svm, y_pred, train_time, pred_time


def train_isolation_forest(X_train, X_test):
    """
    Train Isolation Forest and predict anomalies on the test set.
    """
    start_time = time()  # Record training start time
    model = IsolationForest(random_state=42)
    model.fit(X_train)
    train_time = time() - start_time  # Calculate training time

    # Predict anomalies: -1 -> anomaly, 1 -> normal
    pred_start_time = time()
    y_pred = model.predict(X_test)
    pred_time = time() - pred_start_time  # Calculate prediction time

    return model, y_pred, train_time, pred_time


def train_elliptic_envelope(X_train, X_test):
    """
    Train Elliptic Envelope and predict anomalies on the test set.
    """
    start_train = time()  # Correctly calls time()
    model = EllipticEnvelope()
    model.fit(X_train)
    train_time = time() - start_train  # Training time

    # Predict: -1 -> anomaly, 1 -> normal
    start_pred = time()
    y_pred = model.predict(X_test)
    pred_time = time() - start_pred  # Prediction time

    return model, y_pred, train_time, pred_time


def train_local_outlier_factor(X_train, X_test):
    """
    Train Local Outlier Factor and predict anomalies on the test set.
    """
    start_train = time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_train)  # Train with LOF
    train_time = time() - start_train

    start_pred = time()
    y_pred = lof.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time() - start_pred

    return lof, y_pred, train_time, pred_time


def train_hdbscan(X_train, X_test, min_samples=5, min_cluster_size=10, outlier_threshold=0.9):
    """
    Train HDBSCAN for anomaly detection using clustering and outlier scores.
    """
    # Training HDBSCAN
    start_time = time()
    model = hdbscan.HDBSCAN(min_samples=min_samples,
                            min_cluster_size=min_cluster_size,
                            prediction_data=True)
    model.fit(X_train)
    train_time = time() - start_time

    # Predicting on test data using outlier scores
    pred_start_time = time()
    outlier_scores = hdbscan.approximate_predict(
        model, X_test)[1]  # Get outlier scores
    y_pred = np.array(
        [-1 if score > outlier_threshold else 1 for score in outlier_scores])
    pred_time = time() - pred_start_time

    return model, y_pred, train_time, pred_time
