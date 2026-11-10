import tensorflow as tf
from tensorflow.keras import layers, models

def build_image_model(input_shape=(224, 224, 3), weights="imagenet", learning_rate=1e-4, fine_tune_at=None):
    """
    Builds a MobileNetV2-based model for deepfake & AI-generated image detection.
    Maintains exact architecture compatibility:
    Base -> GlobalAveragePooling2D -> Dropout -> Dense(1, sigmoid).
    Supports two-stage transfer learning:
    - Stage 1: fine_tune_at=None (frozen backbone)
    - Stage 2: fine_tune_at=integer (unfreezes backbone from layer index onwards)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # Binary classification: 0=fake, 1=real
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
