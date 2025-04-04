import os
import sys
import time
import zipfile
import joblib
import warnings
import argparse
import tracemalloc
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.exceptions import UndefinedMetricWarning

import CIC_IoT_Dataset_2023
import ACI_IoT_Dataset_2023
from utils import applyPCA, evaluate_model
from models import train_isolation_forest, train_elliptic_envelope, train_lof, train_one_class_svm

# Ensure current directory is in path
sys.path.append(os.path.dirname(__file__))

# Suppress specific warnings from none issues with EE
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train_models(data_dir, output_dir):
    """
    Trains unsupervised anomaly detection models (Elliptic Envelope, Isolation Forest,
    Local Outlier Factor, and One-Class SVM) using 5-fold cross-validation.

    Each fold uses one file as test data and the rest as training data.
    PCA is applied to reduce dimensionality before model training.
    Trained models and PCA transformers are saved to disk and zipped at the end.

    Parameters:
        data_dir (str): Path to the directory containing Merged01.csv through Merged05.csv.
        output_dir (str): Path where models and PCA objects will be saved.

    Example usage:
        train_models("./data", "./trained_models")
    """
    files = [os.path.join(data_dir, f"Merged{i:02d}.csv") for i in range(1, 6)]
    os.makedirs(output_dir, exist_ok=True)

    for i, test_file in enumerate(files):
        print(f"\n=== Fold {i+1}: Testing on {test_file} ===")

        # Load scaled data (PCA is based on training fold)
        X_train_scaled, y_train, X_test_scaled, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
            test_file)

        # Apply PCA
        start_pca = time.time()
        X_train, X_test, pca = applyPCA(X_train_scaled, X_test_scaled)
        pca_time = (time.time() - start_pca) * 1000
        print(f"PCA took {pca_time:.2f} ms")
        joblib.dump(pca, os.path.join(output_dir, f"pca_fold_{i+1}.pkl"))

        # Define training function template
        model_defs = [
            ("Elliptic Envelope", train_elliptic_envelope,
             f"elliptic_envelope_fold_{i+1}.pkl"),
            ("Isolation Forest", train_isolation_forest,
             f"isolation_forest_fold_{i+1}.pkl"),
            ("LOF", train_lof, f"lof_fold_{i+1}.pkl"),
            ("One-Class SVM", train_one_class_svm,
             f"svm_fold_{i+1}.pkl"),
        ]

        for name, train_func, model_filename in model_defs:
            print(f"\nTraining {name}...")
            start_train = time.time()
            model, y_pred, scores = train_func(X_train, X_test)
            train_time = (time.time() - start_train) * 1000
            print(f"{name} training and prediction took {train_time:.2f} ms")

            evaluate_model(y_test, y_pred, scores,
                           model_name=f"{name} Fold {i+1}")
            joblib.dump(model, os.path.join(output_dir, model_filename))

    # Zip all models into a single archive
    zip_path = os.path.join(output_dir, "trained_models_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".pkl"):
                    full_path = os.path.join(root, f)
                    arcname = os.path.relpath(full_path, output_dir)
                    zipf.write(full_path, arcname)

    print(f"\nModels zipped to: {zip_path}")


def test_models(data_dir, model_dir):
    """
    Tests unsupervised anomaly detection models across 5 cross-validation folds.

    For each fold (Merged01.csv to Merged05.csv):
    - Loads the corresponding test dataset and PCA model
    - Applies PCA transformation (timed and memory-profiled)
    - Loads trained models (Elliptic Envelope, Isolation Forest, LOF, One-Class SVM)
    - Applies decision_function to get anomaly scores (timed and memory-profiled)
    - Uses thresholding on scores (15th percentile) to classify anomalies
    - Evaluates performance using precision, recall, F1, and ROC-AUC

    Parameters:
        data_dir (str): Path to directory containing Merged01–05.csv files
        model_dir (str): Path to directory containing trained models and PCA files

    Example:
        test_models("./data", "./trained_models")
    """
    for fold in range(1, 6):
        print(f"\n=== Fold {fold}: Testing on Merged{fold:02d}.csv ===")

        # Load test dataset
        test_file = os.path.join(data_dir, f"Merged{fold:02d}.csv")
        X_train_scaled, y_train, X_test_scaled, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
            test_file)

        # Load PCA model and transform test data
        pca_path = os.path.join(model_dir, f"pca_fold_{fold}.pkl")
        pca = joblib.load(pca_path)

        tracemalloc.start()
        start_pca = time.time()
        X_test = pca.transform(X_test_scaled)
        pca_time = (time.time() - start_pca) * 1000  # in ms
        _, peak_pca = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"PCA transformation took {pca_time:.2f} ms | Peak Memory: {peak_pca / 1024:.2f} KB\n")

        # Define model files to load
        models = {
            "Elliptic Envelope": f"elliptic_envelope_fold_{fold}.pkl",
            "Isolation Forest": f"isolation_forest_fold_{fold}.pkl",
            "LOF": f"lof_fold_{fold}.pkl",
            "One-Class SVM": f"svm_fold_{fold}.pkl"
        }

        # Evaluate each model
        for name, filename in models.items():
            print(f"--- Evaluating {name} ---")
            model_path = os.path.join(model_dir, filename)
            model = joblib.load(model_path)

            tracemalloc.start()
            start_pred = time.time()
            scores = model.decision_function(X_test)
            pred_time = (time.time() - start_pred) * 1000  # in ms
            _, peak_pred = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(
                f"{name} prediction took {pred_time:.2f} ms | Peak Memory: {peak_pred / 1024:.2f} KB")

            # Use 15th percentile threshold to flag anomalies
            threshold = np.percentile(scores, 15)
            preds = (scores < threshold).astype(int)

            evaluate_model(y_test, preds, scores, model_name=name)
            print()


def main():
    """
    Entry point for training or testing unsupervised network anomaly detection models.

    Usage Examples:
        Train mode:
            python source/main.py --mode train --data_dir ./data --model_dir ./trained_models

        Test mode:
            python source/main.py --mode test --data_dir ./data --model_dir ./trained_models
    """
    parser = argparse.ArgumentParser(
        description="Train or test unsupervised IoT anomaly detection models."
    )
    parser.add_argument(
        '--mode', choices=['train', 'test'], required=True,
        help="Mode to run the program in: 'train' to train models, 'test' to evaluate them."
    )
    parser.add_argument(
        '--data_dir', required=True,
        help="Path to the dataset directory containing the CSV files."
    )
    parser.add_argument(
        '--model_dir', default="./trained_models",
        help="Directory to save/load trained models and PCA files."
    )

    args = parser.parse_args()

    if args.mode == 'train':
        train_models(args.data_dir, args.model_dir)
    elif args.mode == 'test':
        test_models(args.data_dir, args.model_dir)
    else:
        print("Invalid mode selected.")


if __name__ == "__main__":
    main()
