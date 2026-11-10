import tensorflow as tf
from tensorflow.keras import layers, models

def build_video_model(num_frames=10, image_size=(224, 224, 3), weights="imagenet"):
    """Builds a TimeDistributed + Bidirectional GRU model for video deepfake detection."""
    base_mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=image_size,
        include_top=False,
        weights=weights
    )
    spatial_extractor = tf.keras.Sequential([
        base_mobilenet,
        layers.GlobalAveragePooling2D()
    ], name="spatial_feature_extractor")

    # Video input shape: (batch, frames, height, width, channels)
    video_input = layers.Input(shape=(num_frames,) + image_size, name="video_frames_input")
    
    # Apply spatial feature extraction to each frame
    x = layers.TimeDistributed(spatial_extractor, name="time_distributed_spatial")(video_input)
    
    # Process temporal sequence with Bidirectional GRU
    x = layers.Bidirectional(layers.GRU(32, dropout=0.3, return_sequences=False))(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability_real_output")(x)

    model = models.Model(inputs=video_input, outputs=output, name="UnifiedVideoDeepfakeDetector_V2")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
