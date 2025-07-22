import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

base_dir = '/content/drive/MyDrive/neykuri'
train_dir = f"{base_dir}/train"
test_dir = f"{base_dir}/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 20 #adjust if needed

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_dir,
    image_size=IMG_SIZE,
    label_mode="categorical",
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_dir,
    image_size=IMG_SIZE,
    label_mode="categorical"
)

class_names = train_data.class_names
print("Class names:", class_names)

def normalize_img(image, label):
    return image / 255., label

normalized_train_data = train_data.map(normalize_img)
normalized_test_data = test_data.map(normalize_img)

AUTOTUNE = tf.data.experimental.AUTOTUNE
normalized_train_data = normalized_train_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_test_data = normalized_test_data.cache().prefetch(buffer_size=AUTOTUNE)

#using DenseNet121 as base model
from tensorflow.keras.applications import DenseNet121

base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  #freeze layers

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  #output layer
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train model on normalized data
history = model.fit(normalized_train_data, epochs=25)

#accuracy before fine-tuning
test_loss, test_acc = model.evaluate(normalized_test_data)
print(f'\nTest Accuracy: {test_acc * 100:.2f}%')

#fine-tuning
base_model.trainable = True  #unfreeze DenseNet layers

#recompile with lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train for 7 epochs
history_finetune = model.fit(normalized_train_data, epochs=10)

#accuracy after fine-tuning
test_loss, test_acc = model.evaluate(normalized_test_data)
print(f'\nTest Accuracy After Fine-Tuning: {test_acc * 100:.2f}%')

model_path = '/content/densenet121.h5'
model.save(model_path)

#for backend
import shutil
shutil.make_archive('/content/densenet121', 'zip', '/', 'content/densenet121.h5')
from google.colab import files
files.download('/content/densenet121.zip')

#Predictions and visualizations
test_images, test_labels = next(iter(normalized_test_data))
predictions = model.predict(test_images)
predicted_classes = tf.argmax(predictions, axis=1)
true_classes = tf.argmax(test_labels, axis=1)

def plot_predictions(images, true_labels, predicted_labels, class_names, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"Actual: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_labels[i]]}")
        plt.axis('off')
    plt.show()

plot_predictions(test_images, true_classes.numpy(), predicted_classes.numpy(), class_names)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history_finetune.history['accuracy'], label='Fine-Tune Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history_finetune.history['loss'], label='Fine-Tune Loss')
plt.legend()
plt.title('Loss')
plt.show()

#Confusion Matrix
y_true = []
y_pred = []

for images, labels in normalized_test_data:
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(model.predict(images), axis=1).numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap="viridis", ax=plt.gca())
plt.title("Confusion Matrix")
plt.show()

#ROC Curve
y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
n_classes = y_true_binarized.shape[1]

y_pred_proba = model.predict(normalized_test_data)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap('tab10', n_classes)

for i, color in zip(range(n_classes), colors(range(n_classes))):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Class")
plt.legend(loc="lower right")
plt.grid()
plt.show()