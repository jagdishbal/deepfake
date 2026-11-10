import tensorflow as tf
from tensorflow.keras import layers, models

def build_audio_model(input_shape=(140,)):
    """Builds the MLP architecture for audio deepfake detection."""
    model = models.Sequential([
        layers.Input(shape=input_shape, name="acoustic_features_input"),
        layers.Normalization(axis=-1, name="acoustic_norm"),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', name="probability_real_output")
    ], name="AudioDeepfakeDetector_V2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
