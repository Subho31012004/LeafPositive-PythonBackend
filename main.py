import tensorflow as tf
from keras import layers, models, applications, callbacks
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Hyperparameters and dataset path
batch_size = 50
img_height = 300
img_width = 300
dataset_path = "potato_dataset"  # Change this to "dataset_wheat" for the wheat dataset
epochs = 50
validation_split = 0.2  # 20% of data will be used for validation

# Check dataset path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {os.path.abspath(dataset_path)}")

# Load dataset with automatic train-validation split
full_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset="training"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset="validation"
)

# Get class names
class_names = full_ds.class_names
num_classes = len(class_names)
print(f"Class names: {class_names}")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2)
])

# Apply data augmentation to training dataset
full_ds = full_ds.map(lambda x, y: (data_augmentation(x), y))

# Load pre-trained ResNet50 model (without top layers)
base_model = applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # Freeze base layers initially

# Add custom CNN layers on top of ResNet
inputs = layers.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

# Create model
model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callbacks
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train only the custom CNN layers first
history = model.fit(
    full_ds, validation_data=val_ds, epochs=5, callbacks=[lr_scheduler, early_stopping]
)

# Fine-tune ResNet layers (Unfreeze top layers)
base_model.trainable = True
for layer in base_model.layers[:140]:  # Freeze first 140 layers, fine-tune last ones
    layer.trainable = False

# Recompile model with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train again with fine-tuned ResNet layers
history_finetune = model.fit(
    full_ds, validation_data=val_ds, epochs=epochs, callbacks=[lr_scheduler, early_stopping]
)

# Evaluate model
val_loss, val_acc = model.evaluate(val_ds)
print(f"Test Accuracy: {val_acc:.4f}")

# Save model
dataset_name = os.path.basename(dataset_path)  # Extract dataset name for dynamic model naming
model.save(f"potato2_resnet50_finetuned.h5")

# Plot training accuracy & loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_finetune.history['accuracy'], label='Training Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history_finetune.history['loss'], label='Training Loss')
plt.plot(history_finetune.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
