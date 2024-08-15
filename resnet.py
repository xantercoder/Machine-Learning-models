import tensorflow as tf
from tensorflow.keras import mixed_precision
import seaborn as sns
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

mixed_precision.set_global_policy('mixed_float16')

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
IMAGE_SIZE = [640, 640]

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

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    trainMyImagesFolder,
    target_size=(640, 640),
    batch_size=32,
    class_mode='binary',  # For binary classification
    subset='training'  # Set subset to 'training'
)

val_generator = datagen.flow_from_directory(
    trainMyImagesFolder,
    target_size=(640, 640),
    batch_size=32,
    class_mode='binary',  # For binary classification
    subset='validation'  # Set subset to 'validation'
)

# Load ResNet50 model
myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
print(myResnet.summary())

# Freeze the weights
for layer in myResnet.layers:
    layer.trainable = False

# Build the model
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)
# Add the last layer
predictionLayer = Dense(1, activation='sigmoid', dtype='float32')(PlusFlattenLayer)
# Use sigmoid for binary classification

model = Model(inputs=myResnet.input, outputs=predictionLayer)
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

# Callbacks for saving the best model, reducing learning rate, and early stopping
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(results_dir, 'Forest_Fire_Detection_ResNet50.keras'),
                                       verbose=1,
                                       save_best_only=True,
                                       monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
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
plt.savefig(os.path.join(results_dir, 'resnet50_accuracy_plot.png'))

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(results_dir, 'resnet50_loss_plot.png'))

plt.show()

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(
    testMyImagesFolder,
    target_size=(640, 640),
    batch_size=64,
    class_mode='binary'  # For binary classification
)

y_val = test_set.classes
y_pred = model.predict(test_set)
y_pred = np.round(y_pred).astype(int).flatten()  # Use np.round for binary classification

# Print classification report
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true=y_val, y_pred=y_pred)

# Define labels for the confusion matrix
cm_plot_labels = ['fire', 'nofire']

# Plot and save the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_plot_labels, yticklabels=cm_plot_labels)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(results_dir, 'resnet50_confusion_matrix_seaborn.png'))
plt.show()

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot and save the normalized confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=cm_plot_labels, yticklabels=cm_plot_labels)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(results_dir, 'resnet50_normalized_confusion_matrix_seaborn.png'))
plt.show()
