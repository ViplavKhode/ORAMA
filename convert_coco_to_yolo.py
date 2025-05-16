import json
import os
import shutil
import cv2

def convert_bdd_to_yolo(label_dir, output_dir, image_dir):
    # Create label directory
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_output_dir, exist_ok=True)

    # Create image directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # BDD100K detection classes
    category_map = {
        "person": 0,
        "rider": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "train": 5,
        "motorcycle": 6,
        "bicycle": 7,
        "traffic light": 8,
        "traffic sign": 9
    }
    class_names = list(category_map.keys())

    # Process each image in the image directory
    images_with_labels = 0
    images_without_labels = 0
    total_images = 0

    print(f"Processing directory: {image_dir}")
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(".jpg"):
            print(f"Skipping non-JPG file: {img_name}")
            continue
        
        total_images += 1
        # Corresponding label file
        label_file = img_name.replace(".jpg", ".json")
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f"Label file not found for {img_name} at {label_path}")
            images_without_labels += 1
            continue

        # Load the JSON file
        with open(label_path, "r") as f:
            data = json.load(f)

        # Get image dimensions dynamically
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_name}")
            images_without_labels += 1
            continue
        img_height, img_width = img.shape[:2]

        # Get annotations from frames[0]["objects"]
        frames = data.get("frames", [])
        if not frames:
            print(f"No frames found for {img_name}")
            images_without_labels += 1
            continue

        # Assuming single frame (frames[0])
        annotations = frames[0].get("objects", [])
        if not annotations:
            print(f"No objects found for {img_name}")
            images_without_labels += 1
            continue

        # Collect valid annotations
        valid_annotations = []
        for ann in annotations:
            if "box2d" not in ann:
                print(f"Skipping annotation without box2d in {img_name}: {ann['category']}")
                continue
            class_name = ann["category"]
            class_id = category_map.get(class_name, -1)
            if class_id == -1:
                print(f"Unsupported class '{class_name}' in {img_name}")
                continue
            bbox = ann["box2d"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            valid_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Only create a label file and copy the image if there are valid annotations
        if valid_annotations:
            # Write label file
            label_file = os.path.join(label_output_dir, img_name.replace(".jpg", ".txt"))
            with open(label_file, "w") as f:
                f.writelines(valid_annotations)
            # Copy image
            src_path = os.path.join(image_dir, img_name)
            dst_path = os.path.join(image_output_dir, img_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied image {img_name} to {image_output_dir}")
            images_with_labels += 1
            print(f"Created label file for {img_name} with {len(valid_annotations)} annotations")
        else:
            print(f"No valid annotations for {img_name} after filtering")
            images_without_labels += 1

    print(f"Processed {total_images} images: {images_with_labels} with labels, {images_without_labels} without labels")
    return class_names

# Convert train, val, and test sets
train_label_dir = "./bdd100k_labels/100k/train"
val_label_dir = "./bdd100k_labels/100k/val"
train_image_dir = "./filtered/train"
val_image_dir = "./filtered/val"
test_image_dir = "./filtered/test"

os.makedirs("./yolo_format/train", exist_ok=True)
os.makedirs("./yolo_format/val", exist_ok=True)
os.makedirs("./yolo_format/test", exist_ok=True)

print("Converting training set...")
class_names = convert_bdd_to_yolo(train_label_dir, "./yolo_format/train", train_image_dir)
print("\nConverting validation set...")
convert_bdd_to_yolo(val_label_dir, ".yolo_format/val", val_image_dir)
print("\nConverting test set...")
convert_bdd_to_yolo(val_label_dir, "./yolo_format/test", test_image_dir)

# Create data.yaml
with open("C:./yolo_format/data.yaml", "w") as f:
    f.write(f"""
train: ./yolo_format/train/images
val: ./yolo_format/val/images
test: ./yolo_format/test/images
nc: {len(class_names)}
names: {class_names}
""")