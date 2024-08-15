import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

# Ensure GPU is being used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define image size
IMAGE_SIZE = [300, 300]  # EfficientNetB3 standard input size

# Define paths
trainMyImagesFolder = "D:/DATASETS/Resnet/Training"
testMyImagesFolder = "D:/DATASETS/Resnet/Testing"
results_dir = "D:/results"

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

# Create an ImageDataGenerator for data augmentation and splitting
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of the data for validation
)

# Reduce batch size to help with GPU memory usage
batch_size = 16

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    trainMyImagesFolder,
    target_size=(300, 300),  # Adjusted for EfficientNetB3
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='training'  # Set subset to 'training'
)

val_generator = datagen.flow_from_directory(
    trainMyImagesFolder,
    target_size=(300, 300),  # Adjusted for EfficientNetB3
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='validation'  # Set subset to 'validation'
)

# Load EfficientNetB3 model
base_model = EfficientNetB3(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
print(base_model.summary())

# Freeze the weights
for layer in base_model.layers:
    layer.trainable = False

# Build the model
global_avg_pooling_layer = GlobalAveragePooling2D()(base_model.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)

# Add the last layer for binary classification
predictionLayer = Dense(1, activation='sigmoid')(PlusFlattenLayer)  # Sigmoid for binary classification

model = Model(inputs=base_model.input, outputs=predictionLayer)
print(model.summary())

model.compile(loss='binary_crossentropy',  # Binary cross-entropy for binary classification
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Reduced learning rate for EfficientNetB3
              metrics=['accuracy'])

# Custom callbacks to handle serialization issues with EagerTensors
class CustomReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert any TensorFlow tensors in logs to float before serialization
        for key, value in logs.items():
            if isinstance(value, tf.Tensor):
                logs[key] = float(value)
        super().on_epoch_end(epoch, logs)

class SafeModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert TensorFlow tensors to Python floats for serialization
        for key, value in logs.items():
            if isinstance(value, tf.Tensor):
                logs[key] = float(value)
        super().on_epoch_end(epoch, logs)

# Callbacks for saving the best model, reducing learning rate, and early stopping
callbacks = [
    SafeModelCheckpoint(os.path.join(results_dir, 'ForestFire_detection_EfficientNetB3.keras'),
                        verbose=1,
                        save_best_only=True,
                        monitor='val_accuracy'),
    CustomReduceLROnPlateau(monitor='val_accuracy',
                            patience=10,
                            factor=0.1,
                            verbose=1,
                            min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                     patience=30,
                                     verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=75,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# Print the best validation accuracy
best_val_acc = max(history.history['val_accuracy'])
print(f"Best validation Accuracy : {best_val_acc}")

# Save training history graphs
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(results_dir, 'efficientnetb3_accuracy_plot.png'))

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(results_dir, 'efficientnetb3_loss_plot.png'))

plt.show()

# Evaluate the model on the validation set
y_val = val_generator.classes
y_pred = model.predict(val_generator)
y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary class predictions

# Classification report
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true=y_val, y_pred=y_pred)

# Plot confusion matrix using seaborn
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap='Blues'):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Define labels for the confusion matrix
cm_plot_labels = ['fire', 'nofire']

# Plot and save the confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'efficientnetb3_confusion_matrix.png'))

# Plot and save the normalized confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, normalize=True, title='Normalized Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'efficientnetb3_normalized_confusion_matrix.png'))
