# Required Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers, Sequential

# Set random seed for reproducibility
keras.utils.set_random_seed(2211)

# Load and process dataset
example_set = keras.utils.image_dataset_from_directory(
    "draw_example",
    label_mode="categorical",
    color_mode="grayscale",
    shuffle=False,
    batch_size=None,
    image_size=(28,28),
)

# Display a sample image and label
for image, label in example_set:
    print('Image shape:', image.shape, 'Label shape:', label.shape)
    break

# Visualize first 9 training images
plt.figure(figsize=(10, 10))
for i in range(9):
    image = image[i, :, :, 0]
    label = label[i, :]
    label_text = 'Unknown' if np.sum(label) != 1 else y_names[np.argwhere(label == 1)[0, 0]]
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(image, cmap="gray")
    ax.set_title(label_text)
    ax.set_axis_off()
plt.tight_layout()

# Data reshaping and encoding
x_train, x_val, x_test = pa2.reshape_x(x_train_raw, x_val_raw, x_test_raw)
y_train, y_val, y_test = pa2.encode_y(y_train_raw, y_val_raw, y_test_raw, N_labels)

# Augmentation demonstration
augmentation_layer = pa2.AugmentationLayer()
plt.figure(figsize=(10, 10))
image = x_train[0, :, :, :]
label = y_train[0]
label_text = 'Unknown' if np.sum(label) != 1 else y_names[np.argwhere(label == 1)[0, 0]]
plt.suptitle(label_text + " Augmentation")
plt.subplot(3, 3, 1)
plt.imshow(image[:, :, 0], cmap="gray")
plt.axis(False)
for i in range(1, 9):
    ax = plt.subplot(3, 3, i + 1)
    augim = augmentation_layer(image, training=True)
    ax.imshow(augim[:, :, 0].numpy(), cmap="gray")
    ax.set_axis_off()
plt.tight_layout()

# Build, compile, and train model
model = pa2.build_model(N_labels)
pa2.compile_model(model, 0.001)
history = pa2.train_model(model, 20, x_train, y_train, x_val, y_val)

# Evaluate the model
pa2.evaluate_model(model, x_test, y_test)

# Save model
model.save("draw_model.h5")

# Function to reshape the dataset
def reshape_x(x_train_raw, x_val_raw, x_test_raw):
    """Reshape images to proper shapes."""
    return x_train_raw.reshape(28000, 28, 28, 1), x_val_raw.reshape(6000, 28, 28, 1), x_test_raw.reshape(6000, 28, 28, 1)

# Function to one-hot encode labels
def encode_y(y_train_raw, y_val_raw, y_test_raw, N_labels):
    """One-hot encode the labels."""
    return tf.one_hot(y_train_raw, N_labels), tf.one_hot(y_val_raw, N_labels), tf.one_hot(y_test_raw, N_labels)

# Data augmentation layer
def AugmentationLayer():
    """Creates a keras model for data augmentation."""
    return Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation((-0.1, 0.1), fill_mode="constant", interpolation="bilinear", fill_value=0.0)
    ])

# Model building function
def build_model(N_labels):
    """Creates and returns a model."""
    model = Sequential([
        AugmentationLayer(),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(N_labels, activation='softmax')
    ])
    model.build((None, 28, 28, 1))
    return model

# Compile model
def compile_model(model, lr):
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=lr), metrics=["accuracy"])

# Train model
def train_model(model, epochs, x_train, y_train, x_val, y_val):
    return model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val))

# Evaluate model
def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, batch_size=32)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

# Predict images
def predict_images(model, x):
    predictions = model.predict(x)
    return np.argmax(predictions, axis=1)

