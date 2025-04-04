import time
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
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
    model = IsolationForest(contamination=0.15, random_state=42)
        
    train_start = time.time()
    model.fit(X_train)
    train_end = time.time()
    print(f"Training took {train_end - train_start:.4f}s")

    # Generate decision scores
    decision_scores = model.decision_function(X_test)

    # Threshold-based predictions (customizable threshold)
    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_end = time.time()
    print(f"Prediction took {pred_end - pred_start:.4f}s")  
    
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

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
    # contamination=0.15 - Indicates that 15% of the data is considered anomalies (outliers) for fitting the Gaussian distribution.
    # support_fraction=0.80 - Indicates that 80% of the data is considered inliers (normal points) for fitting the Gaussian distribution.
    ee = EllipticEnvelope(
        contamination=0.15, support_fraction=0.8, random_state=42)
    
    train_start = time.time()
    ee.fit(X_train)
    train_end = time.time()
    print(f"Training took {train_end - train_start:.4f}s")

    # Generate decision scores
    decision_scores = ee.decision_function(X_test)

    # Use a relaxed threshold
    threshold = np.percentile(decision_scores, 15)  # Top 15% anomalies

    pred_start = time.time()
    y_pred = ee.predict(X_test)
    y_pred = (decision_scores < threshold).astype(int)
    pred_end = time.time()
    print(f"Prediction took {pred_end - pred_start:.4f}s") 

    return ee, y_pred, decision_scores


def visualize_elliptic_envelope(X_train, X_test, model, decision_scores, n_components=2):
    """
    Visualize the Elliptic Envelope with the decision ellipse after dimensionality reduction.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        model: Trained Elliptic Envelope model.
        decision_scores (array-like): Decision function scores for the test set.
        n_components (int): Number of components for PCA.
    """
    # Reduce to 2 dimensions for visualization
    pca = PCA(n_components=n_components)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Train a new Elliptic Envelope on reduced data for visualization
    vis_model = EllipticEnvelope(
        contamination=0.15, support_fraction=0.8, random_state=42)
    vis_model.fit(X_train_2d)

    # Create grid for visualization
    xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 500),
                         np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 500))
    Z = vis_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=decision_scores,
                cmap='coolwarm', s=50, edgecolor='k', label='Test Data')
    plt.colorbar(label='Decision Score')
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                color='green', s=20, label='Training Data', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                colors='red')  # Decision ellipse

    plt.title("Elliptic Envelope - Outlier Detection with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


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
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.15, novelty=True)
    train_start = time.time()
    lof.fit(X_train)
    train_end = time.time()
    print(f"Training took {train_end - train_start:.4f}s")

    # Generate decision scores
    decision_scores = lof.decision_function(X_test)

    # Threshold: bottom 15% scores = outliers
    threshold = np.percentile(decision_scores, 15)  
    
    pred_start = time.time()
    y_pred = lof.predict(X_test)
    y_pred = (decision_scores < threshold).astype(int)
    pred_end = time.time()
    print(f"Prediction took {pred_end - pred_start:.4f}s") 

    return lof, y_pred, decision_scores


def train_one_class_svm(X_train, X_test):
    """
    Train One-Class SVM and make predictions.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.

    Returns:
        svm: Trained One-Class SVM model.
        y_pred: Binary predictions for the test set.
        decision_scores: Decision function scores for the test set.
    """
    svm = OneClassSVM(kernel="rbf", gamma='scale', nu=0.01)  # Adjust `gamma` and `nu` as needed
    train_start = time.time()
    svm.fit(X_train)
    train_end = time.time()
    print(f"Training took {train_end - train_start:.4f}s")

    # Generate decision scores (distance to the decision boundary)
    decision_scores = svm.decision_function(X_test)

    # Predict anomalies (-1 for anomaly, 1 for normal)
    pred_start = time.time()
    y_pred = svm.predict(X_test)
    pred_end = time.time()
    print(f"Prediction took {pred_end - pred_start:.4f}s") 

    # Convert to binary format: 1 for anomaly, 0 for normal
    y_pred = np.where(y_pred == -1, 1, 0)

    return svm, y_pred, decision_scores
