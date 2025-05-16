import os
import shutil
from sklearn.model_selection import train_test_split

# Paths for training set
train_image_dir = "C:/Aaris/bdd100k_images/100k/train"
train_output_dir = "C:/Aaris/Orama/BDD100K/filtered/train"
os.makedirs(train_output_dir, exist_ok=True)

# Copy all training images (no filtering)
train_images = [img_name for img_name in os.listdir(train_image_dir) if img_name.endswith(".jpg")]
for img_name in train_images:
    src_path = os.path.join(train_image_dir, img_name)
    dst_path = os.path.join(train_output_dir, img_name)
    shutil.copy(src_path, dst_path)
    print(f"Copied {img_name} to filtered/train")

# Paths for validation set (will be split into val and test)
val_image_dir = "C:/Aaris/bdd100k_images/100k/val"
val_output_dir = "C:/Aaris/Orama/BDD100K/filtered/val"
test_output_dir = "C:/Aaris/Orama/BDD100K/filtered/test"
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Copy all validation images and split into val and test
val_images_all = [img_name for img_name in os.listdir(val_image_dir) if img_name.endswith(".jpg")]
val_images, test_images = train_test_split(val_images_all, test_size=0.33, random_state=42)

# Copy validation images
for img_name in val_images:
    src_path = os.path.join(val_image_dir, img_name)
    dst_path = os.path.join(val_output_dir, img_name)
    shutil.copy(src_path, dst_path)
    print(f"Copied {img_name} to filtered/val")

# Copy test images
for img_name in test_images:
    src_path = os.path.join(val_image_dir, img_name)
    dst_path = os.path.join(test_output_dir, img_name)
    shutil.copy(src_path, dst_path)
    print(f"Copied {img_name} to filtered/test")

print(f"Copied {len(train_images)} training images, {len(val_images)} validation images, and {len(test_images)} test images.")