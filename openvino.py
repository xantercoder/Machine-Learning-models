import itertools

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = [640, 640]
trainMyImagesFolder = "D:/DATASETS/Resnet/train"
testMyImagesFolder = "D:/DATASETS/Resnet/validation"

myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
print(myResnet.summary())

# freeze the weights
for layer in myResnet.layers:
    layer.trainable = False

Classes = glob('D:/Resnet/train/*')
numOfClasses = len(Classes)

# build the model
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)
# add the last layer
predictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)

model = Model(inputs=myResnet.input, outputs=predictionLayer)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

# data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(trainMyImagesFolder, target_size=(640, 640),
                                                 batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory(testMyImagesFolder, target_size=(640, 640),
                                            batch_size=32, class_mode='categorical')

EPOCHS = 50
best_model_file = 'D:/results/Crack_detection.keras'

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)]

# train
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=callbacks)

# print the best validation accuracy
best_val_acc = max(history.history['val_accuracy'])
print(f"Best validation Accuracy : {best_val_acc}")

# plot the results / history

plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
y_val = test_set.classes
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_true=y_val, y_pred=y_pred)


# Define function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the plot as PNG or JPEG
    plt.savefig('confusion_matrix.png')  # Change the file name and extension as desired


# Define labels for the confusion matrix
cm_plot_labels = ['Collapsed', 'Major-Crack', 'Minor-crack']

# Call the function to plot the confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
