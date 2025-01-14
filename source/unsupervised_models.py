import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train_isolation_forest(X_train, X_test):
    """
    Train Isolation Forest and make predictions.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.

    Returns:
        model: Trained Isolation Forest model.
        y_pred: Binary predictions for the test set.
        decision_scores: Decision function scores for the test set.
    """
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)

    # Generate decision scores
    decision_scores = model.decision_function(X_test)

    # Threshold-based predictions (customizable threshold)
    threshold = np.percentile(decision_scores, 5)  # Top 5% anomalies
    y_pred = (decision_scores < threshold).astype(int)  # Invert for anomalies

    return model, y_pred, decision_scores


def train_elliptic_envelope(X_train, X_test):
    """
    Train Elliptic Envelope and make predictions.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.

    Returns:
        model: Trained Elliptic Envelope model.
        y_pred: Binary predictions for the test set.
        decision_scores: Decision function scores for the test set.
    """
    model = EllipticEnvelope(
        contamination=0.1, support_fraction=0.9, random_state=42)
    model.fit(X_train)

    # Generate decision scores
    decision_scores = model.decision_function(X_test)

    # Threshold-based predictions (customizable threshold)
    threshold = np.percentile(decision_scores, 5)  # Top 5% anomalies
    y_pred = (decision_scores < threshold).astype(int)  # Invert for anomalies

    return model, y_pred, decision_scores


def train_lof(X_train, X_test):
    """
    Train Local Outlier Factor (LOF) and make predictions.

    Parameters:
        X_train (array-like): Training features (required for fitting).
        X_test (array-like): Test features.

    Returns:
        model: Trained Local Outlier Factor model.
        y_pred: Binary predictions for the test set.
        decision_scores: Decision function scores for the test set.
    """
    # Initialize LOF with novelty detection enabled
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_train)

    # Generate decision scores
    decision_scores = -lof.decision_function(X_test)  # Negate for consistency

    # Threshold-based predictions (customizable threshold)
    threshold = np.percentile(decision_scores, 95)  # Top 5% anomalies
    y_pred = (decision_scores > threshold).astype(int)

    return lof, y_pred, decision_scores


def train_one_class_svm(X_train, X_test):
    """
    Train One-Class SVM and make predictions.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.

    Returns:
        model: Trained One-Class SVM model.
        y_pred: Binary predictions for the test set.
        decision_scores: Decision function scores for the test set.
    """
    model = OneClassSVM(kernel="rbf", gamma=0.1,
                        nu=0.05)  # Adjust `gamma` and `nu` as needed
    model.fit(X_train)

    # Generate decision scores (distance to the decision boundary)
    decision_scores = model.decision_function(X_test)

    # Predict anomalies (-1 for anomaly, 1 for normal)
    y_pred = model.predict(X_test)
    # Convert to binary format: 1 for anomaly, 0 for normal
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    return model, y_pred, decision_scores
