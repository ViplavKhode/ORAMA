from ultralytics import YOLO
import cv2
import numpy as np
import logging
from datetime import datetime
import time
import os
import argparse

# Parse command-line arguments for debug mode
parser = argparse.ArgumentParser(description="Inference script for ORAMA project")
parser.add_argument("--debug", action="store_true", help="Enable debug mode to show processing metrics")
args = parser.parse_args()

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"output/inference_log_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Weather detection (simplified heuristic)
def detect_weather(frame):
    """Detect weather condition based on frame brightness (simplified)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 50:
        return "night"
    return "fog"

# Preprocessing functions for different weather conditions
def preprocess_fog(frame):
    """Dehaze the frame for fog or rain conditions."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(gray, np.ones((5, 5), np.uint8))
        atmospheric_light = np.percentile(dark_channel, 90)
        transmission = 1 - 0.75 * (dark_channel / (atmospheric_light + 1e-6))
        transmission = np.clip(transmission, 0.2, 1)
        dehazed = frame.copy().astype(float)
        for c in range(3):
            dehazed[:, :, c] = (dehazed[:, :, c] - atmospheric_light) / (transmission + 1e-6) + atmospheric_light
        dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
        return dehazed
    except Exception as e:
        logger.error(f"Error in fog preprocessing: {str(e)}")
        return frame

def preprocess_night(frame):
    """Adjust brightness and contrast for night conditions."""
    try:
        alpha = 1.5  # Contrast control
        beta = 50    # Brightness control
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted
    except Exception as e:
        logger.error(f"Error in night preprocessing: {str(e)}")
        return frame

# Estimate distance (heuristic based on bounding box size)
def estimate_distance(box_height, frame_height):
    reference_height = 1.5  # meters (average car height)
    focal_length = frame_height  # Simplified assumption
    distance = (reference_height * focal_length) / box_height
    return max(1, min(50, distance))

# Determine direction of the object
def determine_direction(x_center, frame_width):
    frame_center = frame_width / 2
    if x_center < frame_center - frame_width * 0.1:
        return "left"
    elif x_center > frame_center + frame_width * 0.1:
        return "right"
    else:
        return "center"

# Risk assessment with distance, direction, and percentage
def assess_risk(results, frame_height, frame_width):
    try:
        risk_info = []
        collision_classes = ['car', 'truck', 'motorcycle', 'person']
        for r in results[0].boxes:
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            box_height = y2 - y1
            x_center = (x1 + x2) / 2
            class_id = int(r.cls[0])
            class_name = results[0].names[class_id]
            if class_name in collision_classes and box_height > 0.3 * frame_height:
                distance = estimate_distance(box_height, frame_height)
                direction = determine_direction(x_center, frame_width)
                # Risk percentage: 100% at 0m, 0% at 50m
                risk_percentage = max(0, min(100, (50 - distance) / 50 * 100))
                risk_info.append({
                    "class_name": class_name,
                    "distance": distance,
                    "direction": direction,
                    "risk_percentage": risk_percentage,
                    "box": (x1, y1, x2, y2)
                })
        return risk_info
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        return []

# Function to draw text with a background using OpenCV
def draw_text_with_background(img, text, position, font, font_scale, text_color, thickness, bg_color, border_color=None, margin_top=5, alpha=0.5):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    rect_x1 = x - 2
    rect_y1 = y - text_height - baseline - 2 - margin_top
    rect_x2 = x + text_width + 2
    rect_y2 = y + baseline + 2
    overlay = img.copy()
    if border_color:
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), border_color, 1)
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

# Main loop
try:
    # Load the model
    model = YOLO("runs/train/weights/best.pt")
    logger.info("Model loaded successfully")

    # Load the video
    cap = cv2.VideoCapture("./input/video.mp4")
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video file")
    logger.info("Video loaded successfully")

    # Set frame dimensions
    frame_height = 320
    frame_width = 480
    combined_width = frame_width * 2
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up video writer with timestamp in filename
    output_path = f"output/output_inference_{timestamp}.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (combined_width, frame_height))
    logger.info(f"Output video will be saved to: {output_path}")

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 12 / 30  # 12px
    alert_font_scale = 16 / 30  # 16px
    info_font_scale = 10 / 30   # 10px
    title_color = (255, 255, 255)  # White
    alert_color = (0, 0, 255)      # Red for collision risk
    info_color = (255, 255, 0)     # Yellow
    bg_color = (0, 0, 0)           # Black background
    border_color = (255, 255, 255) # White border for alerts
    thickness = 1

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Measure processing time
        start_time = time.time()

        # Log progress every 100 frames
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")

        # Detect weather condition
        weather = detect_weather(frame)

        # Apply appropriate preprocessing
        if weather == "fog":
            processed_frame = preprocess_fog(frame)
        elif weather == "night":
            processed_frame = preprocess_night(frame)
        else:
            processed_frame = frame  # Default to no preprocessing

        # Perform object detection
        results_pre = model(processed_frame, conf=0.4)

        # Log performance metrics
        logger.info(f"Speed: {results_pre[0].speed['preprocess']:.1f}ms preprocess, {results_pre[0].speed['inference']:.1f}ms inference, {results_pre[0].speed['postprocess']:.1f}ms postprocess")

        # Print class names (run once to verify)
        if frame_count == 1:
            logger.info(f"Model class names: {results_pre[0].names}")

        # Filter out traffic signs and traffic lights
        collision_classes = ['car', 'truck', 'motorcycle', 'person']
        filtered_boxes = [box for box in results_pre[0].boxes if results_pre[0].names[int(box.cls[0])] in collision_classes]
        results_pre[0].boxes = filtered_boxes

        # Assess collision risk
        risk_info = assess_risk(results_pre, frame_height, frame_width)

        # Draw bounding boxes manually with risk percentage on risky objects
        annotated_pre = processed_frame.copy()
        for box in results_pre[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            is_risky = any(r["box"] == (x1, y1, x2, y2) for r in risk_info)
            color = (0, 0, 255) if is_risky else (0, 255, 0)  # Red for risky, green for others
            cv2.rectangle(annotated_pre, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if is_risky:
                # Find the risk percentage for this box
                risk_data = next(r for r in risk_info if r["box"] == (x1, y1, x2, y2))
                risk_percentage = risk_data["risk_percentage"]
                risk_text = f"Risk: {risk_percentage:.1f}%"
                # Draw the risk percentage above the bounding box
                draw_text_with_background(
                    annotated_pre, risk_text, (int(x1), int(y1) - 10), font, info_font_scale,
                    alert_color, thickness, bg_color, margin_top=5, alpha=0.5
                )

        # Combine raw and processed frames side by side
        combined = np.hstack((frame, annotated_pre))

        # Add titles
        draw_text_with_background(combined, "Raw Video", (10, 15), font, title_font_scale, title_color, thickness, bg_color, margin_top=5, alpha=0.5)
        draw_text_with_background(combined, "Processed (Object Detection)", (frame_width + 10, 15), font, title_font_scale, title_color, thickness, bg_color, margin_top=5, alpha=0.5)

        # Add watermark
        draw_text_with_background(combined, "Orama", (combined_width - 60, 15), font, info_font_scale, title_color, thickness, bg_color, margin_top=5, alpha=0.5)

        # Add frame counter (no timestamp)
        draw_text_with_background(combined, f"Frame: {frame_count}/{total_frames}", (10, 35), font, info_font_scale, info_color, thickness, bg_color, margin_top=5, alpha=0.5)

        # Add processing time (debug mode only)
        if args.debug:
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # in milliseconds
            status_text = f"Processing: {processing_time:.1f}ms"
            draw_text_with_background(combined, status_text, (frame_width + 10, frame_height - 15), font, info_font_scale, info_color, thickness, bg_color, margin_top=5, alpha=0.5)

        # Write and display the frame
        out.write(combined)
        cv2.imshow("Orama", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    logger.error(f"Error in main loop: {str(e)}")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Processing complete, resources released")