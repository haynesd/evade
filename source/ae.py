import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def applyAutoencoder(X_train_scaled, X_test_scaled, encoding_dim=10, epochs=20, batch_size=32):
    """
    Trains an autoencoder and applies dimensionality reduction.

    Parameters:
        X_train_scaled (ndarray): Scaled training features.
        X_test_scaled (ndarray): Scaled test features.
        encoding_dim (int): Size of encoded representation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        X_train_encoded (DataFrame): Encoded training data.
        X_test_encoded (DataFrame): Encoded test data.
    """
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(X_train_scaled, X_train_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test_scaled, X_test_scaled),
                    verbose=0)

    X_train_encoded = pd.DataFrame(encoder.predict(X_train_scaled))
    X_test_encoded = pd.DataFrame(encoder.predict(X_test_scaled))

    print(f"Autoencoder reduced dimensions to: {encoding_dim}")

    return X_train_encoded, X_test_encoded
