import tensorflow as tf
import matplotlib.pyplot as plt
import os

# List of model files and their corresponding dataset paths
model_dataset_pairs = {
    "Models/rice_resnet50_finetuned.h5": "datasets/good_dataset",
    "Models/potato2_resnet50_finetuned.h5": "datasets/potato_dataset",
    "Models/wheat2_resnet50_finetuned.h5": "datasets/Wheat_leaf_dataset",
    "Models/tomato_resnet50_finetuned.h5": "datasets/tomato_dataset1",
    "Models/cauliflower_resnet50_finetuned.h5": "datasets/cauliflower_dataset"
}

# Define dataset parameters
img_height = 300
img_width = 300
batch_size = 50

# Store model names and accuracies
model_names = []
val_accuracies = []

# Evaluate each model on its corresponding dataset
for model_file, dataset_path in model_dataset_pairs.items():
    if not os.path.exists(model_file):
        print(f"Warning: Model file {model_file} not found. Skipping...")
        continue
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} not found. Skipping...")
        continue

    print(f"Evaluating model: {model_file} on dataset: {dataset_path}")

    # Load test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Load model and evaluate
    model = tf.keras.models.load_model(model_file)
    _, accuracy = model.evaluate(test_ds, verbose=0)
    
    model_names.append(model_file.replace("all_resnet50_finetuned.h5", ""))
    val_accuracies.append(accuracy)

# Plot model accuracy comparison
plt.figure(figsize=(10, 6))
#plt.bar(model_names, val_accuracies, color=['r', 'g', 'b', 'c', 'm'])
plt.bar(['Rice', 'Potato', 'Wheat', 'Tomato', 'Cauliflower'], val_accuracies, color=['r', 'g', 'b', 'c', 'm'])
plt.xlabel('Model')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison of Models')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with accuracy values
for i, acc in enumerate(val_accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center', fontsize=12)

plt.show()
