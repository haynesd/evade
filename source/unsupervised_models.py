import numpy as np
import time
import hdbscan
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def train_one_class_svm(X_train, X_test):
    """
    Train One-Class Support Vector Machine (SVM) and predict anomalies on the test set.
    """
    start_train = time.time()
    one_class_svm = OneClassSVM(gamma='auto')
    one_class_svm.fit(X_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = one_class_svm.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time.time() - start_pred

    return one_class_svm, y_pred, train_time, pred_time


def train_isolation_forest(X_train, X_test):
    """
    Train Isolation Forest and predict anomalies on the test set.
    """
    start_train = time.time()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = iso_forest.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time.time() - start_pred

    return iso_forest, y_pred, train_time, pred_time


def train_elliptic_envelope(X_train, X_test):
    """
    Train Elliptic Envelope and predict anomalies on the test set.
    """
    start_train = time.time()
    elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
    elliptic_env.fit(X_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = elliptic_env.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time.time() - start_pred

    return elliptic_env, y_pred, train_time, pred_time


def train_local_outlier_factor(X_train, X_test):
    """
    Train Local Outlier Factor and predict anomalies on the test set.
    """
    start_train = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_train)  # Train with LOF
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = lof.predict(X_test)  # -1 for anomalies, 1 for inliers
    pred_time = time.time() - start_pred

    return lof, y_pred, train_time, pred_time


def train_hdbscan(X_train, X_test, min_samples=5, min_cluster_size=10):
    """
    Train and evaluate HDBSCAN on the test set.
    """
    if X_test is None or X_test.shape[0] == 0:
        raise ValueError("X_test is empty or None. HDBSCAN cannot proceed.")

    start_train = time.time()
    hdbscan_model = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
    hdbscan_model.fit(X_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    cluster_labels = hdbscan_model.fit_predict(X_test)  # -1 for noise, cluster labels otherwise
    pred_time = time.time() - start_pred

    # Convert -1 to 0 for anomalies, all other labels to 1
    y_pred = np.array([0 if label == -1 else 1 for label in cluster_labels])

    return hdbscan_model, y_pred, train_time, pred_time


# Utility function to calculate metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy and print the confusion matrix for predictions.
    """
    # Include all possible labels (0 for anomalies, 1 for normal)
    labels = [0, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Generate confusion matrix with specified labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"Confusion Matrix:\n{cm}")

    return accuracy
