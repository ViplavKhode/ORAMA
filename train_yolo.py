from ultralytics import YOLO
import os

def train_model():
    try:
        # Load the model
        model = YOLO("yolov8n.pt")

        model.train(
            data="./yolo_format/data.yaml",
            epochs=70, # Number of epochs
            imgsz=640, # Image size
            batch=16, # Batch size
            device=0, # GPU index
            augment=True,  # Enable data augmentation
            hsv_h=0.015,  # Hue augmentation
            hsv_s=0.7,  # Saturation augmentation (can simulate fog/rain effects)
            hsv_v=0.4,  # Value (brightness) augmentation (can simulate fog/rain effects)
            mosaic=1.0,  # Mosaic augmentation
            mixup=0.2,  # Mixup augmentation
            fliplr=0.5,  # Horizontal flip
            translate=0.1,  # Translation
            scale=0.5,  # Scaling
            shear=0.1,  # Shearing
            perspective=0.0001  # Perspective transform
        )

        # Validate the model
        metrics = model.val()
        print(f"mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    train_model()