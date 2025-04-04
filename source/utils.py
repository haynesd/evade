import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def applyPCA(X_train_scaled, X_test_scaled):
    start_time = time.time()
    pca = PCA()
    pca.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Selected PCA components: {n_components}")
    pca = PCA(n_components=n_components)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train_scaled))
    X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))
    print(f"PCA took {time.time() - start_time:.2f} seconds")
    return X_train_pca, X_test_pca, pca


def evaluate_model(y_true, y_pred, scores, model_name="Model"):
    try:
        roc_auc = roc_auc_score(y_true, -scores)
    except:
        roc_auc = 0.0
    print(f"\n{model_name} Results:")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
