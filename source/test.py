import supervised_models
import unsupervised_models
import data_preparation
import data_processing
import visualization
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Todo: Fix Elliptical 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_test():
    base_dir = os.getcwd()  # Current working directory
    data_folder = os.path.join(base_dir, "nad", "data")  # Path to /nad/data folder
    csv_file = os.path.join(data_folder, "ACI-IoT-2023-Payload.csv")  # Path to test.csv file

    X_train, y_train, X_test, y_test = data_processing.load_and_preprocess_data(csv_file)

    # Check shapes before calling the model training
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Store the metrics for visualization
    model_names = []
    accuracies = []
    train_times = []
    pred_times = []

    # Train and evaluate the Random Forest (supervised)
    # Train and evaluate the Random Forest
    print("Training Random Forest...")
    rf_model, rf_acc, rf_train_time, rf_pred_time = supervised_models.train_random_forest(X_train, y_train, X_test, y_test)
    # Print the result
    print(f"Random Forest Accuracy: {rf_acc:.2f}")
    print(f"Random Forest Training Time: {rf_train_time:.2f} seconds, Prediction Time: {rf_pred_time:.2f} seconds")
    # Append Random Forest results
    model_names.append("Random Forest")
    accuracies.append(rf_acc)
    train_times.append(rf_train_time)
    pred_times.append(rf_pred_time)

    # Train and evaluate Isolation Forest (one-class classifier unsupervised)
    print("Training Isolation Forest...")
    iso_forest_model, iso_forest_acc, iso_forest_train_time, iso_forest_pred_time = unsupervised_models.train_isolation_forest(X_train, X_test)
    print(f"Isolation Forest Accuracy: {iso_forest_acc:.2f}")
    model_names.append("Isolation Forest")
    accuracies.append(iso_forest_acc)
    train_times.append(iso_forest_train_time)
    pred_times.append(iso_forest_pred_time)

    # Train and evaluate Elliptic Envelope (one-class classifier unsupervised)
    print("Training Elliptic Envelope...")
    elliptic_env_model, elliptic_env_acc, elliptic_env_train_time, elliptic_env_pred_time = unsupervised_models.train_elliptic_envelope(X_train, X_test)
    print(f"Elliptic Envelope Accuracy: {elliptic_env_acc:.2f}")
    model_names.append("Elliptic Envelope")
    accuracies.append(elliptic_env_acc)
    train_times.append(elliptic_env_train_time)
    pred_times.append(elliptic_env_pred_time)

    # Train and evaluate Local Outlier Factor (LOF) (one-class classifier unsupervised)
    print("Training LOF...")
    lof_model, lof_acc, lof_train_time, lof_pred_time = unsupervised_models.train_local_outlier_factor(X_train, X_test)
    print(f"LOF Accuracy: {lof_acc:.2f}")
    model_names.append("LOF")
    accuracies.append(lof_acc)
    train_times.append(lof_train_time)  # LOF does not have a training time
    pred_times.append(lof_pred_time)

    # Visualize the results 
    visualization.visualize_results(model_names, accuracies, train_times, pred_times)


if __name__ == "__main__":
    run_test()

    # SVM Test take over 3 hours
    # Train and evaluate the SVM
    # print("Training SVM...")
    # svm_model, svm_acc, svm_auc, svm_train_time, svm_pred_time = supervised_models.train_svm(X_train, y_train, X_test, y_test)
    # print(f"SVM Accuracy: {svm_acc:.2f}, AUC: {svm_auc:.2f}")
    # print(f"SVM Training Time: {svm_train_time:.2f} seconds, Prediction Time: {svm_pred_time:.2f} seconds")

    # # Append SVM results
    # model_names.append("SVM")
    # accuracies.append(svm_acc)
    # aucs.append(svm_auc)
    # train_times.append(svm_train_time)
    # pred_times.append(svm_pred_time)

    # Train and evaluate the One-Class SVM (one-class classifier unsupervised)
    # print("Training OneClassSVM...")
    # one_class_svm_model, one_class_svm_acc, one_class_train_time, one_class_pred_time = unsupervised_models.train_one_class_svm(X_train, X_test)
    # print(f"OneClassSVM Accuracy: {one_class_svm_acc:.2f}")
    # print(f"OneClassSVM Training Time: {one_class_train_time:.2f} seconds, Prediction Time: {one_class_pred_time:.2f} seconds")

    # # Append One-Class SVM results
    # model_names.append("OneClassSVM")
    # accuracies.append(one_class_svm_acc)
    # aucs.append(0)  # No AUC for unsupervised
    # train_times.append(one_class_train_time)
    # pred_times.append(one_class_pred_time)

