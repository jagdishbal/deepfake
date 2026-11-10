"""
V3 Dual-Branch Spatial-Frequency Residual Model Architecture and Layer Definitions.
Maintains exact serialization compatibility with best_model_v3.keras.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def get_srm_kernels():
    """
    Constructs the 3 standard SRM (Spatial Rich Model) residual filters.
    All filters are 5x5 with sum = 0 to capture high-frequency noise residuals.
    """
    # Filter 1: 1st order residual
    q1 = np.zeros((5, 5), dtype=np.float32)
    q1[2, 2] = -1.0
    q1[2, 3] = 1.0
    
    # Filter 2: 2nd order residual
    q2 = np.zeros((5, 5), dtype=np.float32)
    q2[2, 1] = 1.0
    q2[2, 2] = -2.0
    q2[2, 3] = 1.0
    
    # Filter 3: 3rd order SQUARE residual
    q3 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0
    
    lum_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    kernel_weights = np.zeros((5, 5, 3, 3), dtype=np.float32)
    for c in range(3):
        kernel_weights[:, :, c, 0] = q1 * lum_weights[c]
        kernel_weights[:, :, c, 1] = q2 * lum_weights[c]
        kernel_weights[:, :, c, 2] = q3 * lum_weights[c]
        
    return kernel_weights

@tf.keras.utils.register_keras_serializable(package="test_v3_model", name="SRMResidualLayer")
class SRMResidualLayer(layers.Layer):
    """
    Fixed spatial-rich-model residual filter layer with reflection padding.
    Extracts high-frequency noise residuals without learning weights.
    """
    def __init__(self, **kwargs):
        super(SRMResidualLayer, self).__init__(**kwargs)
        self.kernel_weights = get_srm_kernels()
        
    def build(self, input_shape):
        self.srm_kernel = self.add_weight(
            name="srm_fixed_kernel",
            shape=(5, 5, 3, 3),
            initializer=tf.keras.initializers.Constant(self.kernel_weights),
            trainable=False
        )
        super(SRMResidualLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Reflection pad of 2 pixels on each side for 5x5 convolution
        padded = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        # Apply 2D convolution with fixed SRM filter
        residuals = tf.nn.conv2d(padded, self.srm_kernel, strides=[1, 1, 1, 1], padding="VALID")
        return tf.clip_by_value(residuals, -2.0, 2.0)
        
    def get_config(self):
        config = super(SRMResidualLayer, self).get_config()
        return config


def load_v3_image_model(checkpoint_path: str | Path):
    """
    Safely loads the trained V3 Keras model with custom SRMResidualLayer registered.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"V3 checkpoint not found at: {checkpoint_path}")
        
    custom_objects = {
        "SRMResidualLayer": SRMResidualLayer,
        "test_v3_model>SRMResidualLayer": SRMResidualLayer
    }
    return tf.keras.models.load_model(str(checkpoint_path), custom_objects=custom_objects, compile=False)
