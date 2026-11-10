import tensorflow as tf
from pathlib import Path

def preprocess_image_tensor(image: tf.Tensor, target_size: tuple = (224, 224)) -> tf.Tensor:
    """
    Preprocesses an RGB image tensor without distorting aspect ratios:
    1. Pads symmetrically to preserve aspect ratio (no squashing).
    2. Applies MobileNetV2 normalization [-1, 1].
    """
    if len(image.shape) == 2:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
        
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1], method="bilinear")
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def get_image_dataset(directory: str | Path, batch_size: int = 32, image_size: tuple = (224, 224), subset: str = None, shuffle: bool = True):
    """
    Creates a tf.data.Dataset from a directory of images.
    Preserves aspect ratios and applies normalization and data augmentation if for training.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    # Using crop_to_aspect_ratio=True avoids squashing/stretching non-square images
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dir_path),
        labels='inferred',
        label_mode='binary',
        class_names=['fake', 'real'],  # explicitly define to ensure consistency: fake=0, real=1
        batch_size=batch_size,
        image_size=image_size,
        crop_to_aspect_ratio=True,
        shuffle=shuffle,
        seed=42 if shuffle else None,
    )
    
    # Apply augmentations to [0, 255] pixels for training
    if subset == 'train':
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(
        lambda x, y: (
            tf.keras.applications.mobilenet_v2.preprocess_input(x),
            y,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
