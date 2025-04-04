import zipfile
import unsupervised_models
import CIC_IoT_Dataset_2023
import ACI_IoT_Dataset_2023
from utils import applyPCA, evaluate_model
from sklearn.decomposition import PCA
import os
import argparse
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))


def train_models(data_dir, output_dir):
    files = [os.path.join(data_dir, f"Merged{i:02d}.csv") for i in range(1, 6)]
    os.makedirs(output_dir, exist_ok=True)

    for i, test_file in enumerate(files):
        print(f"\n=== Fold {i+1}: Testing on {test_file} ===")
        train_files = [f for j, f in enumerate(files) if j != i]

        # Load scaled data
        X_train_scaled, y_train, X_test_scaled, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
            test_file)

        # Apply PCA
        X_train, X_test, pca = applyPCA(X_train_scaled, X_test_scaled)
        joblib.dump(pca, os.path.join(output_dir, f"pca_fold_{i+1}.pkl"))

        # Train and save Elliptic Envelope
        ee_model, ee_y_pred, ee_scores = unsupervised_models.train_elliptic_envelope(
            X_train, X_test)
        evaluate_model(y_test, ee_y_pred, ee_scores,
                       model_name=f"Elliptic Envelope Fold {i+1}")
        joblib.dump(ee_model, os.path.join(
            output_dir, f"elliptic_envelope_fold_{i+1}.pkl"))

        # Train and save Isolation Forest
        iso_model, iso_y_pred, iso_scores = unsupervised_models.train_isolation_forest(
            X_train, X_test)
        evaluate_model(y_test, iso_y_pred, iso_scores,
                       model_name=f"Isolation Forest Fold {i+1}")
        joblib.dump(iso_model, os.path.join(
            output_dir, f"isolation_forest_fold_{i+1}.pkl"))

        # Train and save LOF
        lof_model, lof_y_pred, lof_scores = unsupervised_models.train_lof(
            X_train, X_test)
        evaluate_model(y_test, lof_y_pred, lof_scores,
                       model_name=f"LOF Fold {i+1}")
        joblib.dump(lof_model, os.path.join(output_dir, f"lof_fold_{i+1}.pkl"))

        # Train and save One-Class SVM
        svm_model, svm_y_pred, svm_scores = unsupervised_models.train_one_class_svm(
            X_train, X_test)
        evaluate_model(y_test, svm_y_pred, svm_scores,
                       model_name=f"One-Class SVM Fold {i+1}")
        joblib.dump(svm_model, os.path.join(output_dir, f"svm_fold_{i+1}.pkl"))

    # Zip all models
    zip_path = os.path.join(output_dir, "trained_models_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".pkl"):
                    full_path = os.path.join(root, f)
                    arcname = os.path.relpath(full_path, output_dir)
                    zipf.write(full_path, arcname)
    print(f"\n✅ Models zipped to: {zip_path}")


def test_models(data_dir, model_dir):
    test_file = os.path.join(data_dir, "Merged01.csv")
    X_train_scaled, y_train, X_test_scaled, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
        test_file)

    # Load PCA and apply
    pca = joblib.load(os.path.join(model_dir, "pca_fold_1.pkl"))
    X_test = pca.transform(X_test_scaled)

    # Load and evaluate models
    models = {
        "Elliptic Envelope": "elliptic_envelope_fold_1.pkl",
        "Isolation Forest": "isolation_forest_fold_1.pkl",
        "LOF": "lof_fold_1.pkl",
        "One-Class SVM": "svm_fold_1.pkl"
    }

    for name, filename in models.items():
        path = os.path.join(model_dir, filename)
        model = joblib.load(path)
        scores = model.decision_function(X_test)
        preds = model.predict(X_test)
        evaluate_model(y_test, preds, scores, model_name=name)


def main():
    parser = argparse.ArgumentParser(
        description="Train/Test Unsupervised IoT Anomaly Models")
    parser.add_argument(
        '--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    parser.add_argument('--data_dir', required=True,
                        help="Path to the dataset directory")
    parser.add_argument('--model_dir', default="./trained_models",
                        help="Path to load/save models")

    args = parser.parse_args()

    if args.mode == 'train':
        train_models(args.data_dir, args.model_dir)
    elif args.mode == 'test':
        test_models(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
