import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope


def applyPCA(X_train_scaled, X_test_scaled):
    """
    Fits PCA on the training data to retain 95% variance and applies it to both
    training and test datasets. Returns transformed data and the fitted PCA model.

    Parameters:
        X_train_scaled (np.ndarray or pd.DataFrame): Standardized training data.
        X_test_scaled (np.ndarray or pd.DataFrame): Standardized test data.

    Returns:
        X_train_pca (pd.DataFrame): PCA-transformed training data.
        X_test_pca (pd.DataFrame): PCA-transformed test data.
        pca (PCA): Trained PCA model.

    Example:
        X_train_pca, X_test_pca, pca = applyPCA(X_train_scaled, X_test_scaled)
    """
    start_time = time.time()

    # Determine optimal number of components to retain 95% variance
    pca_temp = PCA()
    pca_temp.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Selected PCA components: {n_components}")

    # Fit PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train_scaled))
    X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))

    elapsed_ms = (time.time() - start_time) * 1000
    print(f"PCA took {elapsed_ms:.2f} ms")

    return X_train_pca, X_test_pca, pca


def evaluate_model(y_true, y_pred, scores, model_name="Model"):
    """
    Evaluates binary classification results using precision, recall, F1 score,
    and ROC-AUC. Outputs results to the console.

    Parameters:
        y_true (array-like): Ground truth binary labels (0 = benign, 1 = anomaly).
        y_pred (array-like): Predicted binary labels (0 = benign, 1 = anomaly).
        scores (array-like): Anomaly scores from decision_function (higher = more normal).
        model_name (str): Name of the model for display.

    Example:
        evaluate_model(y_test, preds, scores, model_name="Isolation Forest")
    """
    try:
        # Flip scores so higher = anomaly
        roc_auc = roc_auc_score(y_true, -scores)
    except Exception:
        roc_auc = 0.0

    print(f"\n{model_name} Results:")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")
    print(f"ROC-AUC:   {roc_auc:.2f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

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

    # Create grid for decision boundary
    xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 500),
                         np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 500))
    Z = vis_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 24})
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=decision_scores,
                          cmap='coolwarm', s=80, edgecolor='k', marker='+', label='Test Data')
    plt.colorbar(scatter, label='Decision Score')

    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                facecolors='none', edgecolors='green', s=70, label='Training Data')

    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

    plt.title("Elliptic Envelope Decision Boundary", fontsize=24)
    #plt.xlabel("PCA Component 1", fontsize=14)
    #plt.ylabel("PCA Component 2", fontsize=14)
    plt.legend(fontsize=22)
    plt.tight_layout()
    #plt.savefig("elliptic_envelope_plot.eps", format='eps')
    #plt.show()

