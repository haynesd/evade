import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import data_processing


def build_autoencoder(input_dim):
    """
    Build an autoencoder model.

    Args:
        input_dim (int): Dimensionality of the input data.

    Returns:
        autoencoder (Model): Compiled autoencoder model.
        encoder (Model): Encoder part of the autoencoder.
    """
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder Model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder


def train_autoencoder(X_train, X_test, epochs=50, batch_size=128):
    """
    Train the autoencoder.

    Args:
        X_train (numpy.ndarray): Training data.
        X_test (numpy.ndarray): Testing data.
        epochs (int): Number of training epochs.
        batch_size (int): Size of the training batches.

    Returns:
        autoencoder (Model): Trained autoencoder model.
        encoder (Model): Encoder part of the trained autoencoder.
        reconstruction_error (numpy.ndarray): Reconstruction error for the test data.
    """
    input_dim = X_train.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)

    # Train the autoencoder
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test),
        verbose=1
    )

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Calculate reconstruction error for test data
    reconstructed = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=1)

    return autoencoder, encoder, reconstruction_error


def detect_anomalies(reconstruction_error, threshold):
    """
    Detect anomalies based on reconstruction error.

    Args:
        reconstruction_error (numpy.ndarray): Reconstruction error for the data.
        threshold (float): Threshold for anomaly detection.

    Returns:
        numpy.ndarray: Binary anomaly labels (1 for anomaly, 0 for normal).
    """
    return (reconstruction_error > threshold).astype(int)


def evaluate_autoencoder(y_test, y_pred):
    """
    Evaluate the autoencoder anomaly detection performance.

    Args:
        y_test (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Predicted anomaly labels.

    Returns:
        None
    """
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")


# Integrate into the pipeline
if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test, feature_names = data_processing.load_and_preprocess_data(
        "data/ACI-IoT-2023-Payload.csv")

    # Train Autoencoder
    autoencoder, encoder, reconstruction_error = train_autoencoder(
        X_train, X_test)

    # Set threshold for anomaly detection
    # Top 5% reconstruction errors are anomalies
    threshold = np.percentile(reconstruction_error, 95)

    # Detect anomalies
    y_pred = detect_anomalies(reconstruction_error, threshold)

    # Evaluate
    evaluate_autoencoder(y_test, y_pred)
