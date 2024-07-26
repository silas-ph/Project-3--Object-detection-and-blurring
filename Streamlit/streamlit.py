import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Define the path to your model using pathlib
model_path = Path("best.pt")

# Load YOLO model
model = YOLO(model_path)

def draw_bounding_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def blur_objects(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Ensure the coordinates are within the frame dimensions
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x1 < x2 and y1 < y2:  # Check if the coordinates are valid
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            frame[y1:y2, x1:x2] = blurred_roi
    return frame

def initialize_tracker(frame, box):
    tracker = cv2.legacy.TrackerCSRT_create()
    x1, y1, x2, y2 = map(int, box)
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return tracker

def update_trackers(frame, trackers):
    boxes = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h
            boxes.append((x1, y1, x2, y2))
    return boxes

def process_video(input_video_path, output_video_path, model, reinit_interval=5):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize variables
    trackers = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % reinit_interval == 0:  # Check for new detections every reinit_interval frames
            results = model(frame)
            detections = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Filter detections by confidence score
            valid_detections = [det for det, conf in zip(detections, confidences) if conf > 0.3]

            # Add new trackers for new detections
            for det in valid_detections:
                if all(np.linalg.norm(det - np.array(box)) > 50 for box in [tracker[1] for tracker in trackers]):  # Avoid duplicate detections
                    tracker = initialize_tracker(frame, det)
                    trackers.append((tracker, det))

        # Update trackers
        updated_boxes = []
        for tracker, box in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                updated_boxes.append((x1, y1, x2, y2))
            else:
                trackers.remove((tracker, box))  # Remove failed trackers

        # Draw bounding boxes for visualization
        frame_with_boxes = draw_bounding_boxes(frame.copy(), updated_boxes)

        # Apply blurring
        frame_with_blur = blur_objects(frame_with_boxes, updated_boxes)

        # Write the frame to the output video
        out.write(frame_with_blur)

        frame_count += 1

    cap.release()
    out.release()

def streamlit_app():
    st.title("Video Processing App")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg4"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = Path(tfile.name)
        
        output_path = video_path.with_suffix('.output.mp4')

        st.write("Loading YOLO model...")
        
        st.write("Processing video...")
        process_video(video_path, str(output_path), model)
        st.write("Video processed.")
        
        with open(output_path, 'rb') as f:
            st.download_button(label="Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")

if __name__ == '__main__':
    streamlit_app()
