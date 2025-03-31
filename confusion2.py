import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# ✅ Update the dataset path correctly
dataset_path = "datasets/potato_dataset"  # Change this if needed

# ✅ Check if the dataset path exists
if not os.path.exists(dataset_path):
    print("❌ Error: Dataset path not found! Check your folder structure.")
    exit()

# ✅ Ensure the image size matches the trained model (300x300)
batch_size = 32
img_size = (300, 300)  # Fix image size issue

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,  # Resizing images to match model input
    batch_size=batch_size,
    shuffle=True  # Shuffle to ensure randomness
)

# ✅ Get class names (disease labels)
class_names = dataset.class_names
print("Detected Classes:", class_names)

# ✅ Load trained model (Make sure this path is correct)
model_path = "Models/potato2_resnet50_finetuned.h5"
model = tf.keras.models.load_model(model_path)

# ✅ Get true labels and predictions
y_true = []
y_pred = []

for images, labels in dataset:
    y_true.extend(labels.numpy())  # Store actual labels
    predictions = model.predict(images)  # Make predictions
    y_pred.extend(np.argmax(predictions, axis=1))  # Convert to class indices

# ✅ Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# ✅ Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# ✅ Compute accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

# ✅ Print results
print(f"\n Model Performance Metrics ")
print(f" Accuracy: {accuracy:.4f}\n")
print(" Classification Report:\n", report)