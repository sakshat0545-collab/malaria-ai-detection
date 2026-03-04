import os
import shutil
import random

# Paths
base_path = "dataset/archive/cell_images"
output_path = "dataset"

parasitized_path = os.path.join(base_path, "Parasitized")
uninfected_path = os.path.join(base_path, "Uninfected")

train_path = os.path.join(output_path, "train")
val_path = os.path.join(output_path, "val")

# Create directories
for category in ["Parasitized", "Uninfected"]:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)

def split_and_copy(src_path, category):
    images = os.listdir(src_path)
    random.shuffle(images)

    selected_images = images[:499]  # Select 499 images
    split_index = int(0.8 * len(selected_images))

    train_images = selected_images[:split_index]
    val_images = selected_images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_path, img),
                    os.path.join(train_path, category, img))

    for img in val_images:
        shutil.copy(os.path.join(src_path, img),
                    os.path.join(val_path, category, img))

# Execute split
split_and_copy(parasitized_path, "Parasitized")
split_and_copy(uninfected_path, "Uninfected")

print("Dataset prepared successfully!")