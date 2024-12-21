
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate_anomaly_detection(y_true, y_pred, model_name, train_time, pred_time):
    """
    Evaluate anomaly detection models using Precision, Recall, F1, and ROC-AUC.
    Handles multiclass outputs for models like HDBSCAN.
    """
    # Map y_true: 0 -> 1 (normal), 1 -> -1 (anomaly)
    y_true_binary = [1 if y == 0 else -1 for y in y_true]

    # For HDBSCAN or multiclass outputs, ensure -1 is treated as the anomaly class
    # Default normal is 1
    y_pred_binary = [-1 if y == -1 else 1 for y in y_pred]

    # Check if there are anomalies (-1) in predictions or ground truth
    if -1 not in y_true_binary or -1 not in y_pred_binary:
        print(
            f"Warning: No anomalies (-1) detected for {model_name}. Metrics set to 0.")
        precision, recall, f1, roc_auc = 0.0, 0.0, 0.0, 0.0
    else:
        # Compute metrics safely
        precision = precision_score(
            y_true_binary, y_pred_binary, pos_label=-1, average='binary', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary,
                              pos_label=-1, average='binary', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, pos_label=-
                      1, average='binary', zero_division=0)
        roc_auc = roc_auc_score([1 if y == -1 else 0 for y in y_true_binary],
                                [1 if y == -1 else 0 for y in y_pred_binary])

    print(f"Metrics for {model_name}:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  ROC-AUC: {roc_auc:.2f}")
    print(f"  Training Time: {train_time:.2f} seconds")
    print(f"  Prediction Time: {pred_time:.2f} seconds\n")

    return precision, recall, f1, roc_auc
