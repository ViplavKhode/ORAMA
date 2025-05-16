<p align="center">
  <img src="https://github.com/user-attachments/assets/dd3dc33f-8fe5-49dc-88be-0eeddd8df78e" />
</p>

# AI-Powered Object Detection and Risk Assessment System

<p align="center">
  <img src="https://github.com/user-attachments/assets/98632123-c84b-4b39-b5d1-aa455635ba59" alt="0323" />
</p>

## Project Overview
ORAMA is an AI-based system designed to enhance vehicle safety in adverse weather conditions, specifically foggy and rainy scenarios. It improves object detection and provides real-time collision risk assessment. Developed as a Minimum Viable Product (MVP) for a hackathon, ORAMA leverages:

- **YOLOv8** for object detection.
- **Weather-adaptive preprocessing** for visibility enhancement.
- **A heuristic-based risk assessment algorithm** to identify potential collisions.

The system processes video input, detects objects, and highlights collision risks with visual alerts, achieving real-time performance at **~29 FPS**.

## Key Features
- **Weather-Adaptive Preprocessing**: Dynamically adjusts for fog and night conditions using dehazing and brightness/contrast enhancement.
- **Object Detection**: Utilizes YOLOv8 to detect collision-relevant objects (cars, trucks, pedestrians) in low-visibility scenarios.
- **Risk Assessment**: Estimates object distance and direction, calculating a risk percentage to identify potential collisions.
- **Visual Alerts**: Highlights risky objects with red bounding boxes and displays the risk percentage (e.g., `Risk: 45.3%`).
- **Real-Time Performance**: Processes frames at ~34.5ms per frame (~29 FPS), suitable for edge deployment.

## Vision
ORAMA aims to become a **Software-as-a-Service (SaaS)** platform for autonomous vehicle manufacturers, providing real-time object detection and risk assessment as a standardized solution. Future plans include:

- **Ethical data collection** from vehicle cameras (with user consent).
- **Integration with additional datasets** for diverse driving conditions.
- **Deployment on high-performance edge devices** for real-world validation.

## Problem Statement
Visibility is significantly reduced in foggy and rainy conditions, making traditional object detection methods unreliable. ORAMA addresses this by:
- Selecting and processing relevant datasets.
- Developing deep learning models for detection.
- Implementing adaptive preprocessing techniques.
- Evaluating performance and risk assessment models.
- Designing real-time alert mechanisms for drivers.

## Repository Structure
```
ORAMA/
├── orama/
│   ├── inference.py              # Main script for inference and risk assessment
│   ├── output/
│   │   ├── inference_log_<timestamp>.log  # Log files
│   │   └── output_inference_<timestamp>.mp4  # Output videos
│   ├── runs/
│   │   └── train3/
│   │       └── weights/
│   │           └── best.pt      # Trained YOLOv8 model weights
│   └── (evaluation outputs: confusion_matrix_normal.png, F1_curve.png, etc.)
├── yolo_train/                   # Directory for training-related files
├── convert_boco_to_yolo.py       # Script to convert BDD100K dataset to YOLO format
├── evaluate.py                   # Script to evaluate model performance
├── filter_bdd100k.py             # Script to filter BDD100K dataset
├── train_yolo.py                 # Script to train YOLOv8 model
├── video.mp4                     # Input video for inference
└── README.md                     # This file
```

## Setup Instructions
### Prerequisites
- **Python 3.8+**
- **Operating System**: Tested on Ubuntu 20.04; should work on Windows and macOS with minor adjustments.
- **Hardware**: GPU recommended for faster inference (e.g., NVIDIA GPU with CUDA support).

### Installation
#### Clone the Repository
```bash
git clone https://github.com/Aaris03Khan/orama.git
cd orama
```

#### Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv orama.env
source orama.env/bin/activate  # On Windows: orama.env\Scripts\activate
```

#### Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

#### Download the Model Weights
- Place the trained YOLOv8 model weights (`best.pt`) in `orama/runs/train3/weights/`.
- If not included, train the model using `train_yolo.py` (see Training section).

#### Prepare the Input Video
- Place your input video as `video.mp4` in the `ORAMA/` directory.
- Alternatively, modify the video path in `inference.py`.

## Usage Instructions
### Running the Inference Script
Navigate to the ORAMA directory:
```bash
cd orama
```

#### Run the script:
```bashRe
python3 inference.py
```

### Viewing the Output
The output video will be saved as:
```bash
output/output_inference_<timestamp>.mp4
```
Logs are saved in `output/inference_log_<timestamp>.log`.

## Results and Performance
### Output Video
- **Left Side**: Raw video with frame counter.
- **Right Side**: Processed video with detected objects (green bounding boxes) and collision risks (red bounding boxes with risk percentage).
<p align="center">
  <img src="https://github.com/user-attachments/assets/efe4e178-5423-4275-9493-de7671eeae53" />
</p>


## Future Work
- **Dataset Expansion**: Add **Foggy Cityscapes, KITTI,** and synthetic data.
- **Preprocessing**: Integrate deep learning-based dehazing and rain removal (e.g., U-Net).
- **Risk Assessment**: Implement **velocity and trajectory analysis** using object tracking (DeepSORT).
- **Alerts**: Introduce **audio alerts and dashboard notifications**.
- **Vehicle Assistance**: Simulate **automatic braking and lane departure warnings**.
- **Edge Deployment**: Optimize inference on **NVIDIA Jetson Nano**.

---
ORAMA is a scalable AI-powered system that enhances road safety in challenging weather conditions. Future developments will focus on improved accuracy, ethical data collection, and seamless integration with autonomous vehicle platforms.
