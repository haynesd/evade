import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


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
