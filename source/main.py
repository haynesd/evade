import os
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
import ACI_IoT_Dataset_2023
import CIC_IoT_Dataset_2023
import unsupervised_models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_model(y_test, y_pred, decision_scores, model_name="Model"):
    """
    Evaluate the model's performance using precision, recall, F1-score, and ROC-AUC.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted binary labels.
        decision_scores (array-like): Decision scores or probabilities for ROC-AUC.
        model_name (str): Name of the model being evaluated.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    # Compute classification metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Compute ROC-AUC using decision scores
    try:
        # Invert if anomalies have lower scores
        roc_auc = roc_auc_score(y_test, -decision_scores)
    except ValueError:
        # Handle cases where ROC-AUC cannot be computed (e.g., single class)
        roc_auc = 0.0

    print(f"\n{model_name} Evaluation:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

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

    end_time = time.time()
    print(f"PCA training and transformation took {end_time - start_time:.4f} seconds")

    return X_train_pca, X_test_pca

def applyAutoencoder(X_train_scaled, X_test_scaled):
    start_time = time.time()

    input_dim = X_train_scaled.shape[1]

    # Define autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(X_train_scaled, X_train_scaled,
                    epochs=20,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test_scaled, X_test_scaled),
                    verbose=0)

    # Encode the data
    encoder = Model(inputs=input_layer, outputs=encoded)
    X_train_encoded = encoder.predict(X_train_scaled)
    X_test_encoded = encoder.predict(X_test_scaled)

    end_time = time.time()
    print(f"Autoencoder training and transformation took {end_time - start_time:.4f} seconds")

    return X_train_encoded, X_test_encoded

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
        X_train_scaled, y_train, X_test_scaled, y_test = CIC_IoT_Dataset_2023.getTrainTestDataFromCSV(
            test_file
        )
        
        X_train, X_test = applyPCA(X_train_scaled, X_test_scaled)

        #X_train, X_test = applyAutoencoder(X_train_scaled, X_test_scaled)
        
        # Train and evaluate Elliptic Envelope
        print("\nTraining Elliptic Envelope...")
        ee_model, ee_y_pred, ee_scores = unsupervised_models.train_elliptic_envelope(
            X_train, X_test
        )

        evaluate_model(y_test, ee_y_pred, decision_scores=ee_scores,
                       model_name=f"Elliptic Envelope Fold {i+1}")

        # Visualize Elliptic Envelope decision boundaries
        unsupervised_models.visualize_elliptic_envelope(
            X_train, X_test, ee_model, ee_scores)

        # Train and evaluate Isolation Forest
        print("\nTraining Isolation Forest...")
        iso_model, iso_y_pred, iso_scores = unsupervised_models.train_isolation_forest(
            X_train, X_test)
        evaluate_model(y_test, iso_y_pred, decision_scores=iso_scores,
                       model_name=f"Isolation Forest Fold {i+1}")

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
                       decision_scores=svm_scores, model_name=f"One-Class SVM Fold {i+1}")


if __name__ == "__main__":
    # Define base directory and data folder

    #base_dir = os.path.dirname(os.getcwd())  # Move up one level from "source"
    #data_folder = os.path.join(base_dir, "data")

    current_dir = os.getcwd()  # Current working directory
    data_folder = os.path.join(current_dir, "data")  # Append 'data' folder    

    # Path to ACI-IoT-2023.csv file
    # aci_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")
    # Train_Test_ACI_IoT_2023(aci_file)

    # Paths to Merged1.csv, Merged2.csv, ..., Merged5.csv files
    cic_files = [os.path.join(
        data_folder, f"Merged{i:02d}.csv") for i in range(1, 6)]
    Train_Test_CIC_IoT_2023(cic_files)

