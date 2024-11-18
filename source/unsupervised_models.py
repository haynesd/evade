import numpy as np
import time
import hdbscan
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def train_one_class_svm(X_train, X_test):
    """
    Train One-Class Suport Vector Machine (SVM) and evaluate the accuracy on the test set.
    This function also measures the time taken for training and prediction. This model constructs 
    a decision boundary that tightly encloses the majority of the training data, which is assumed to be from a 
    "normal" distribution. Points that fall outside this boundary are labeled as anomalies

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Training features.
        X_test (numpy.ndarray or pandas.DataFrame): Test features.

    Returns:
        tuple: Trained One-Class SVM model, accuracy on the test set, and time taken for training and prediction.
    """
    # Start timing
    start_time = time.time()

    # Train the One-Class SVM
    # The kernel coefficient is 'auto' automatically adjusted based on the data. A higher gamma leads to a more complex
    # boundary (more overfitting), while a lower gamma leads to a smoother boundary (more generalization)
    one_class_svm = OneClassSVM(gamma='auto')
    one_class_svm.fit(X_train)

    # Training time
    training_time = time.time() - start_time
    print(f"Training OneClassSVM took {training_time:.2f} seconds")

    # Start timing for prediction
    start_time = time.time()

    # Predict the labels for the test set
    y_pred_test = one_class_svm.predict(X_test)

    # Prediction time
    prediction_time = time.time() - start_time
    print(f"Prediction using OneClassSVM took {prediction_time:.2f} seconds")

    # One-Class SVM returns -1 for outliers and 1 for inliers, convert them to 0 and 1
    y_pred_test = [0 if x == -1 else 1 for x in y_pred_test]

    # Compare against an assumed "normal" class of 1
    # Assuming normal data for unsupervised testing
    y_true = [1] * len(y_pred_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_test)

    return one_class_svm, accuracy, training_time, prediction_time


def train_isolation_forest(X_train, X_test):
    """
    Train an Isolation Forest model and evaluate its accuracy on the test set.
    Also measures the time taken for training and prediction.

    Args:
        X_train (array): Training features.
        X_test (array): Test features.

    Returns:
        tuple: Trained Isolation Forest model, accuracy on the test set, and time taken for training and prediction.
    """
    # Start timing for training
    start_time = time.time()

    # Train the Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(X_train)

    # Training time
    training_time = time.time() - start_time
    print(f"Training Isolation Forest took {training_time:.2f} seconds")

    # Start timing for prediction
    start_time = time.time()

    # Predict the labels for the test set (-1 for outliers, 1 for inliers)
    y_pred_test = iso_forest.predict(X_test)

    # Prediction time
    prediction_time = time.time() - start_time
    print(
        f"Prediction using Isolation Forest took {prediction_time:.2f} seconds")

    # Convert -1 to 0 for outliers and 1 for inliers (so they align with expected labels)
    y_pred_test = [0 if x == -1 else 1 for x in y_pred_test]

    # Since we're assuming the test set is "normal" (i.e., all data points are inliers), create true labels as 1s
    y_true = [1] * len(y_pred_test)

    # Calculate accuracy (comparing predicted labels to all "normal" ground truth labels)
    accuracy = accuracy_score(y_true, y_pred_test)

    return iso_forest, accuracy, training_time, prediction_time


def train_elliptic_envelope(X_train, X_test):
    """
    Train Elliptic Envelope and evaluate the accuracy on the test set.
    Elliptic Envelope fits a multivariate Gaussian distribution to the data and defines an envelope that covers
    most of the data points. Points outside the envelope are considered outliers.

    Args:
        X_train (array): Training features.
        X_test (array): Test features.

    Returns:
        tuple: Trained Elliptic Envelope model, accuracy on the test set, and time taken for training and prediction.
    """
    # Start timing for training
    start_time = time.time()

    # Train the Elliptic Envelope model with higher support_fraction
    elliptic_env = EllipticEnvelope(
        contamination=0.1, support_fraction=0.9, random_state=42)
    elliptic_env.fit(X_train)

    # Training time
    training_time = time.time() - start_time
    print(f"Training Elliptic Envelope took {training_time:.2f} seconds")

    # Start timing for prediction
    start_time = time.time()

    # Predict labels for the test set
    y_pred_test = elliptic_env.predict(X_test)

    # Prediction time
    prediction_time = time.time() - start_time
    print(
        f"Prediction using Elliptic Envelope took {prediction_time:.2f} seconds")

    # Convert -1 to 0 for outliers and 1 for inliers
    y_pred_test = [0 if x == -1 else 1 for x in y_pred_test]

    # Assume the test set is normal (inliers) and create true labels as 1s
    y_true = [1] * len(y_pred_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_test)

    return elliptic_env, accuracy, training_time, prediction_time


def train_local_outlier_factor(X_train, X_test):
    """
    Train Local Outlier Factor and evaluate its accuracy on the test set.
    Local Outlier Factor (LOF) detects the anomaly score based on the local density of the data points.

    Args:
        X_train (array): Training features.
        X_test (array): Test features.

    Returns:
        tuple: LOF model, accuracy on the test set, and time taken for prediction (LOF does not require training).
    """
    # LOF doesn't have a fit method, only a predict method. It is used as an unsupervised outlier detector.
    # Start timing for prediction
    start_time = time.time()

    # Fit and predict using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred_test = lof.fit_predict(X_test)

    # Prediction time
    prediction_time = time.time() - start_time
    print(f"Prediction using LOF took {prediction_time:.2f} seconds")

    # Convert -1 to 0 for outliers and 1 for inliers
    y_pred_test = [0 if x == -1 else 1 for x in y_pred_test]

    # Assume the test set is normal (inliers) and create true labels as 1s
    y_true = [1] * len(y_pred_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_test)

    return lof, accuracy, 0, prediction_time  # LOF does not have a training tim


def train_hdbscan(X_train, X_test, min_samples=5, min_cluster_size=10):
    """
    Train and evaluate HDBSCAN on the test set.
    HDBSCAN is a hierarchical clustering algorithm that adjusts to varying densities and identifies noise points.

    Args:
        X_train (array): Training features (not used for HDBSCAN, as it is unsupervised).
        X_test (array): Test features.
        min_samples (int): Minimum number of samples to define the neighborhood of a point.
        min_cluster_size (int): Minimum size of clusters to be formed.

    Returns:
        tuple: HDBSCAN model, accuracy on the test set, and time taken for prediction.
    """
    # Validate input
    if X_test is None or X_test.shape[0] == 0:
        raise ValueError("X_test is empty or None. HDBSCAN cannot proceed.")

    # Start timing for clustering
    start_time = time.time()

    # Fit HDBSCAN on the test set
    hdbscan_model = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size)
    cluster_labels = hdbscan_model.fit_predict(X_test)

    # Clustering time
    prediction_time = time.time() - start_time
    print(f"Clustering using HDBSCAN took {prediction_time:.2f} seconds")

    # Convert -1 to 0 for noise points and keep other cluster labels as 1 (assuming inliers)
    y_pred_test = [0 if x == -1 else 1 for x in cluster_labels]

    # Assume the test set is normal (inliers) and create true labels as 1s
    y_true = [1] * len(y_pred_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_test)

    # HDBSCAN does not have an explicit training step
    return hdbscan_model, accuracy, 0, prediction_time
