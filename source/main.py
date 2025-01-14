import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
import ACI_IoT_Dataset_2023
import CIC_IoT_Dataset_2023
import unsupervised_models

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_model(y_test, y_pred, decision_scores=None, model_name="Model"):
    """
    Evaluate the model's performance using precision, recall, F1-score, and ROC-AUC.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted binary labels.
        decision_scores (array-like, optional): Decision scores or probabilities for ROC-AUC.
        model_name (str): Name of the model being evaluated.
    """
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if decision_scores is not None:
        roc_auc = roc_auc_score(y_test, decision_scores)
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")


def Train_Test_ACI_IoT_2023(file):
    """
    Train and evaluate unsupervised anomaly detection models on the ACI IoT dataset.

    Parameters:
        file (str): Path to the CSV file containing the dataset.
    """
    # Split into features and target
    X_train, y_train, X_test, y_test = ACI_IoT_Dataset_2023.getTrainTestDataFromCSV(
        file)

    # Train and evaluate Isolation Forest
    print("\nTraining Isolation Forest...")
    iso_model, iso_y_pred, iso_scores = unsupervised_models.train_isolation_forest(
        X_train, X_test)
    evaluate_model(y_test, iso_y_pred, decision_scores=iso_scores,
                   model_name="Isolation Forest")

    # Train and evaluate Elliptic Envelope
    print("\nTraining Elliptic Envelope...")
    ee_model, ee_y_pred, ee_scores = unsupervised_models.train_elliptic_envelope(
        X_train, X_test)
    evaluate_model(y_test, ee_y_pred, decision_scores=ee_scores,
                   model_name="Elliptic Envelope")

    # Train and evaluate LOF
    print("\nTraining LOF...")
    lof_model, lof_y_pred, lof_scores = unsupervised_models.train_lof(
        X_train, X_test)
    evaluate_model(y_test, lof_y_pred,
                   decision_scores=lof_scores, model_name="LOF")


def Train_Test_CIC_IoT_2023(files):
    """
    Perform cross-validation by splitting CIC IoT dataset into train-test folds.

    Parameters:
        files (list): List of file paths for the dataset.
    """
    for i, test_file in enumerate(files):
        train_files = [f for j, f in enumerate(files) if j != i]
        print(f"\n=== Fold {i+1}: Testing on {test_file} ===")
        train_data = pd.concat([pd.read_csv(f)
                               for f in train_files], ignore_index=True)
        test_data = pd.read_csv(test_file)

        # Split into features and target
        X_train, y_train, X_test, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
            test_file)

        # Train and evaluate Isolation Forest
        print("\nTraining Isolation Forest...")
        iso_model, iso_y_pred, iso_scores = unsupervised_models.train_isolation_forest(
            X_train, X_test)
        evaluate_model(y_test, iso_y_pred, decision_scores=iso_scores,
                       model_name=f"Isolation Forest Fold {i+1}")

        # Train and evaluate Elliptic Envelope
        print("\nTraining Elliptic Envelope...")
        ee_model, ee_y_pred, ee_scores = unsupervised_models.train_elliptic_envelope(
            X_train, X_test)
        evaluate_model(y_test, ee_y_pred, decision_scores=ee_scores,
                       model_name=f"Elliptic Envelope Fold {i+1}")

        # Train and evaluate LOF
        print("\nTraining LOF...")
        lof_model, lof_y_pred, lof_scores = unsupervised_models.train_lof(
            X_train, X_test)
        evaluate_model(y_test, lof_y_pred,
                       decision_scores=lof_scores, model_name=f"LOF Fold {i+1}")

        # Train and evaluate One-Class SVM
        print("\nTraining One-Class SVM...")
        svm_model, svm_y_pred, svm_scores = unsupervised_models.train_one_class_svm(
            X_train, X_test)
        evaluate_model(y_test, svm_y_pred,
                       decision_scores=svm_scores, model_name="One-Class SVM")


if __name__ == "__main__":
    # Define base directory and data folder
    base_dir = os.getcwd()
    data_folder = os.path.join(base_dir, "data")

    # Path to ACI-IoT-2023.csv file
    # aci_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")
    # Train_Test_ACI_IoT_2023(aci_file)

    # Paths to Merged1.csv, Merged2.csv, ..., Merged5.csv files
    cic_files = [os.path.join(
        data_folder, f"Merged{i:02d}.csv") for i in range(1, 6)]
    Train_Test_CIC_IoT_2023(cic_files)
