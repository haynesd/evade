import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    This function trains a Random Forest Classifier on a given training 
    dataset (X_train, y_train), and evaluates the classifier's performance on a test 
    dataset (X_test, y_test). It returns the trained model, accuracy, AUC, and the 
    time taken for both training and prediction.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Feature matrix for training.
        y_train (numpy.ndarray or pandas.Series): Labels for training.
        X_test (numpy.ndarray or pandas.DataFrame): Feature matrix for testing.
        y_test (numpy.ndarray or pandas.Series): Labels for testing.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
        float: Accuracy on the test set.
        float: Time taken to train the model (in seconds).
        float: Time taken to predict on the test set (in seconds).
    """


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest Classifier and evaluate its performance.
    Returns the trained model, accuracy, training time, and prediction time.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import time

    # Start timing the training process
    start_time = time.time()

    # Train the model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Record the training time
    rf_train_time = time.time() - start_time

    # Start timing the prediction process
    start_time = time.time()

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Record the prediction time
    rf_pred_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy, rf_train_time, rf_pred_time


def train_svm(X_train, y_train, X_test, y_test):
    """
    This function trains a Support Vector Machine (SVM) classifier and evaluates 
    its performance on a test set. It returns the trained SVM model, accuracy, AUC, 
    and time taken for both training and prediction.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Feature matrix for training.
        y_train (numpy.ndarray or pandas.Series): Labels for training.
        X_test (numpy.ndarray or pandas.DataFrame): Feature matrix for testing.
        y_test (numpy.ndarray or pandas.Series): Labels for testing.

    Returns:
        SVC: Trained SVM model.
        float: Accuracy on the test set.
        float: Time taken to train the model (in seconds).
        float: Time taken to predict on the test set (in seconds).
    """
    # Start timing for training
    start_time = time.time()

    # Train SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    # Training time
    training_time = time.time() - start_time
    print(f"Training SVM took {training_time:.2f} seconds")

    # Start timing for prediction
    start_time = time.time()

    # Predict the labels for the test set
    y_pred_test = svm.predict(X_test)

    # Prediction time
    prediction_time = time.time() - start_time
    print(f"Prediction using SVM took {prediction_time:.2f} seconds")

    # Calculate accuracy and AUC
    accuracy = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

    return svm, accuracy, auc, training_time, prediction_time
