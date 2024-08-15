import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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

# Image size for Inception-ResNet-v2
IMAGE_SIZE = [299, 299]

# Paths to your dataset
train_dir = "D:/DATASETS/Resnet/Training"
results_dir = "D:/results"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Load the Inception-ResNet-v2 model with pre-trained ImageNet weights, excluding the top layer
inception_resnet = InceptionResNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the pre-trained layers
for layer in inception_resnet.layers:
    layer.trainable = False

# Add custom layers on top
x = GlobalAveragePooling2D()(inception_resnet.output)
x = Dropout(0.5)(x)  # Add dropout to reduce overfitting
prediction_layer = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inception_resnet.input, outputs=prediction_layer)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Data augmentation and preprocessing with validation split
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='binary',
    subset='training'  # Set subset to 'training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='binary',
    subset='validation'  # Set subset to 'validation'
)

# Callbacks for saving the best model, reducing learning rate, and early stopping
checkpoint = ModelCheckpoint(
    os.path.join(results_dir, 'forest_fire_detection_inception_resnet_v2.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=10,
    factor=0.1,
    verbose=1,
    min_lr=1e-6
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, early_stopping]

# Training the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=75,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# Best validation accuracy
best_val_acc = max(history.history['val_accuracy'])
print(f"Best Validation Accuracy: {best_val_acc}")

# Save training history graphs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(results_dir, 'loss_plot.png'))

plt.show()

# Evaluate the model on the validation set
y_val = val_generator.classes
y_pred = model.predict(val_generator)
y_pred = np.where(y_pred > 0.5, 1, 0).flatten()

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
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

# Plot and save the normalized confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, normalize=True, title='Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'normalized_confusion_matrix.png'))
