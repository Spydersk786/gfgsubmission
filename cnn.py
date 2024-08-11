import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
irregular_dir = 'synthetic_shapes/irregular'
regular_dir = 'synthetic_shapes/regular'

# ImageDataGenerator for loading and preprocessing
datagen = ImageDataGenerator(rescale=1./255)

# Create a custom generator
def custom_generator(batch_size=32):
    irregular_gen = datagen.flow_from_directory(
        'synthetic_shapes',
        classes=['irregular'],
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,  # No labels are required
        shuffle=True,
        seed=42)

    regular_gen = datagen.flow_from_directory(
        'synthetic_shapes',
        classes=['regular'],
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,  # No labels are required
        shuffle=True,
        seed=42)

    while True:
        irregular_images = next(irregular_gen)
        regular_images = next(regular_gen)
        yield irregular_images, regular_images

# Create the generator
train_generator = custom_generator(batch_size=32)
import tensorflow as tf
from tensorflow.keras import layers, models


import tensorflow as tf
from tensorflow.keras import layers, models

def create_shape_regularization_model():
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(128, 128, 1)))

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))

    # Reshape to image dimensions
    model.add(layers.Dense(8 * 8 * 256, activation='relu'))
    model.add(layers.Reshape((8, 8, 256)))

    # Deconvolution layers (upsampling)
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))
    
    # Final layer: Ensuring the output is 128x128x1
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    return model

# Create the model
model = create_shape_regularization_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Create the model
model = create_shape_regularization_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model using the custom generator
model.fit(train_generator, epochs=30, steps_per_epoch=1000//32, verbose=1)
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.model import load_model

# Load a test image
test_image = cv2.imread('synthetic_shapes/problems/frag1.svg', cv2.IMREAD_GRAYSCALE)
test_image = test_image / 255.0  # Normalize the image
test_image = np.expand_dims(test_image, axis=(0, -1))  # Add batch and channel dimensions

# Predict the regular shape
predicted_regular = model.predict(test_image)

# Display the original and predicted shapes
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Irregular Shape')
plt.imshow(test_image.squeeze(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Predicted Regular Shape')
plt.imshow(predicted_regular.squeeze(), cmap='gray')
plt.show()

