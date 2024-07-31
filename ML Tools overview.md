### Step 4: Processing Video

#### Function Definition and Initial Setup
```python
def process_video(input_video_path, output_video_path, model, reinit_interval=5, progress_callback=None):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
```

- **Function Definition**: `process_video` is defined to handle video processing.
- **Parameters**:
  - `input_video_path`: Path to the input video file.
  - `output_video_path`: Path where the output video will be saved.
  - `model`: The YOLO model used for object detection.
  - `reinit_interval`: Interval at which object detection and tracker initialization occur.
  - `progress_callback`: Optional callback function to update progress.
- **Video Capture**: `cv2.VideoCapture(input_video_path)` opens the input video file.
- **Video Properties**:
  - `width`, `height`: Dimensions of the video frames.
  - `fps`: Frames per second of the video.
  - `total_frames`: Total number of frames in the video.
- **Video Writer**:
  - `fourcc`: Codec for writing the video.
  - `out`: Video writer object to save processed frames to the output file.

#### Initializing Trackers and Frame Counter
```python
    trackers = []
    frame_count = 0
    start_time = time()
```

- **Trackers List**: Initializes an empty list to store tracker objects.
- **Frame Counter**: Sets the initial frame count to 0.
- **Start Time**: Records the start time for tracking processing time.

#### Main Processing Loop
```python
    while True:
        ret, frame = cap.read()
        if not ret:
            break
```

- **Reading Frames**: Reads frames from the video in a loop using `cap.read()`.
  - `ret`: Boolean indicating if the frame was read successfully.
  - `frame`: The current frame of the video.
- **Break Condition**: If `ret` is `False` (end of video or error), the loop breaks.

#### Object Detection and Tracker Initialization
```python
        if frame_count % reinit_interval == 0:
            results = model(frame)
            detections = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            valid_detections = [det for det, conf in zip(detections, confidences) if conf > 0.3]
            for det in valid_detections:
                if all(np.linalg.norm(det - np.array(box)) > 50 for box in [tracker[1] for tracker in trackers]):
                    tracker = initialize_tracker(frame, det)
                    trackers.append((tracker, det))
```

- **Interval Check**: Every `reinit_interval` frames, reinitialize object detection and trackers.
- **YOLO Model Inference**: 
  - `results = model(frame)`: Runs the YOLO model on the current frame.
  - `results[0].boxes.xyxy.cpu().numpy()`: Extracts bounding box coordinates.
  - `results[0].boxes.conf.cpu().numpy()`: Extracts confidence scores.
- **Valid Detections**: Filters detections with confidence scores greater than 0.3.
- **Tracker Initialization**:
  - **Distance Check**: Ensures new detections are not too close to existing tracker boxes.
  - **Initialize Tracker**: Calls `initialize_tracker` for each valid detection.
  - **Append Tracker**: Adds the tracker and its bounding box to the `trackers` list.

#### Updating Trackers
```python
        updated_boxes = []
        for tracker, box in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                updated_boxes.append((x1, y1, x2, y2))
            else:
                trackers.remove((tracker, box))
```

- **Updating Tracker Positions**:
  - **Iterate Trackers**: Loops through each tracker and its associated initial bounding box.
  - **Update Tracker**: Updates the tracker with the current frame.
  - **Bounding Box Conversion**: Converts the updated bounding box to integer coordinates.
  - **Append Updated Boxes**: Adds successfully tracked boxes to `updated_boxes`.
  - **Remove Failed Trackers**: Removes trackers that fail to update.
- **Calculating Coordinates**:
  - **`x1, y1, w, h = map(int, bbox)`**: 
    - Converts the bounding box coordinates to integers. This ensures the pixel values are in whole numbers.
    - `x1, y1`: The top-left corner coordinates of the updated bounding box.
    - `w, h`: The width and height of the updated bounding box.
  - **`x2 = x1 + w`**: Calculates the x-coordinate of the bottom-right corner of the bounding box by adding the width `w` to the x-coordinate `x1` of the top-left corner.
  - **`y2 = y1 + h`**: Calculates the y-coordinate of the bottom-right corner of the bounding box by adding the height `h` to the y-coordinate `y1` of the top-left corner.

#### Blurring Objects and Writing Frames
```python
        frame_with_blur = blur_objects(frame.copy(), updated_boxes)
        out.write(frame_with_blur)
```

- **Blurring Objects**:
  - **Copy Frame**: Creates a copy of the current frame.
  - **Blur Function**: Calls `blur_objects` to blur detected objects in the frame.
- **Write Frame**: Writes the blurred frame to the output video.

#### Frame Counter and Progress Callback
```python
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count / total_frames)
```

- **Increment Frame Counter**: Increments the frame count by 1.
- **Update Progress**: Calls the progress callback function (if provided) with the current progress.

#### Release Resources
```python
    cap.release()
    out.release()
```

- **Release Video Capture**: Closes the video capture object.
- **Release Video Writer**: Closes the video writer object.

### Summary
This function reads frames from the input video, uses the YOLO model to detect objects, initializes and updates trackers, blurs detected objects, and writes the processed frames to an output video file. It also supports progress tracking through a callback function. The coordinates \( x1, y1 \) and \( x2, y2 \) are calculated to define the bounding box for each detected object, ensuring accurate tracking and blurring across frames.



This is just for reference in case I need it.
YOLO (You Only Look Once)
How YOLO Works: YOLO is a real-time object detection system that applies a single neural network to the full image. The network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

Key Steps in YOLO:

Input Image: The image is divided into an ( S \times S ) grid.
Bounding Box Prediction: Each grid cell predicts a fixed number of bounding boxes. Each bounding box consists of four coordinates (x, y, width, height) and a confidence score.
Class Prediction: Each grid cell predicts the probability of each class.
Non-Max Suppression: YOLO applies non-max suppression to reduce the number of overlapping boxes, keeping the most confident ones.
Output: YOLO outputs the bounding boxes with class labels and confidence scores.
Advantages of YOLO:

Speed: YOLO is very fast because it processes the entire image with a single forward pass of the network.
Accuracy: YOLO achieves high accuracy for object detection tasks.
Unified Model: YOLO uses a single network for both classification and localization, simplifying the architecture.
Real World Example: Using YOLO to identify pictures on walls in a dataset of interior room images involves training the YOLO model with annotated images where pictures on the walls are labeled. The trained model can then detect and identify pictures in new images of interior rooms.

Hugging Face
How Hugging Face Works: Hugging Face is a company that provides tools and models for natural language processing (NLP) and computer vision tasks. Their library, transformers, provides pre-trained models for various tasks, including text classification, translation, and image classification.

Key Steps in Using Hugging Face for Image Classification:

Dataset Preparation: Upload and preprocess the dataset of images.
Model Selection: Choose a pre-trained model suitable for image classification from the Hugging Face model hub.
Training: Fine-tune the pre-trained model on the dataset of interior room images with labeled pictures.
Inference: Use the fine-tuned model to predict and identify pictures on walls in new images.
Advantages of Hugging Face:

Pre-trained Models: Hugging Face provides access to a wide range of pre-trained models that can be fine-tuned for specific tasks.
Ease of Use: The transformers library simplifies the process of model training and inference.
Community Support: Hugging Face has a large community of users and contributors, providing extensive documentation and support.
Real World Example: Using Hugging Face to identify pictures on walls in a dataset of interior room images involves fine-tuning a pre-trained model with the labeled dataset. The model can then be used to detect and classify pictures in new images.

Comparison: YOLO vs. Hugging Face
Similarities:

Image Detection and Classification: Both YOLO and Hugging Face can be used to detect and classify objects in images.
Dataset Preparation: Both methods require a dataset of labeled images for training.
Neural Network Models: Both use neural network models for image processing tasks.
Differences:

Primary Focus:

YOLO: Specializes in real-time object detection with high speed and accuracy.
Hugging Face: Primarily focused on NLP but has expanded to include computer vision tasks, providing pre-trained models for various applications.
Model Architecture:

YOLO: Uses a unified model for object detection, which processes the entire image in a single forward pass.
Hugging Face: Provides a range of models, including transformers for vision tasks, which can be fine-tuned for specific datasets.
Ease of Use:

YOLO: Requires more effort to set up and train, but offers high performance for object detection tasks.
Hugging Face: Offers pre-trained models that are easy to fine-tune and integrate, with extensive support and documentation.
Application Speed:

YOLO: Known for its speed and efficiency in real-time object detection.
Hugging Face: May not be as fast as YOLO for real-time detection, but excels in providing versatile and powerful pre-trained models for various tasks.
Business Problem Case: Using images of interior rooms, and then identifying pictures, diplomas or other personally identifying objects. Once identified, then the selected objects will be blurred.
Using YOLO:

Dataset Preparation: Annotate images with bounding boxes around pictures on the walls.
Model Training: Train the YOLO model with the annotated dataset.
Inference: Use the trained YOLO model to detect pictures on walls in new images.
Using Hugging Face:

Dataset Preparation: Annotate images with labels for pictures on the walls.
Model Selection: Choose a pre-trained vision model from Hugging Face.
Model Fine-tuning: Fine-tune the pre-trained model with the annotated dataset.
Inference: Use the fine-tuned model to identify pictures on walls in new images.
Conclusion: Both YOLO and Hugging Face offer powerful tools for image detection and classification tasks. YOLO is ideal for real-time object detection with high speed and accuracy, while Hugging Face provides a versatile platform with pre-trained models that can be easily fine-tuned for specific applications.


### YOLO (You Only Look Once)

**How YOLO Works:**
YOLO is a real-time object detection system that applies a single neural network to the full image. The network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

**Key Steps in YOLO:**
1. **Input Image**: The image is divided into an \( S \times S \) grid.
2. **Bounding Box Prediction**: Each grid cell predicts a fixed number of bounding boxes. Each bounding box consists of four coordinates (x, y, width, height) and a confidence score.
3. **Class Prediction**: Each grid cell predicts the probability of each class.
4. **Non-Max Suppression**: YOLO applies non-max suppression to reduce the number of overlapping boxes, keeping the most confident ones.
5. **Output**: YOLO outputs the bounding boxes with class labels and confidence scores.

**Advantages of YOLO:**
- **Speed**: YOLO is very fast because it processes the entire image with a single forward pass of the network.
- **Accuracy**: YOLO achieves high accuracy for object detection tasks.
- **Unified Model**: YOLO uses a single network for both classification and localization, simplifying the architecture.

**Real World Example**: Using YOLO to identify pictures on walls in a dataset of interior room images involves training the YOLO model with annotated images where pictures on the walls are labeled. The trained model can then detect and identify pictures in new images of interior rooms.



**Differences:**
- **Primary Focus**:
  - **YOLO**: Specializes in real-time object detection with high speed and accuracy.
  - **Hugging Face**: Primarily focused on NLP but has expanded to include computer vision tasks, providing pre-trained models for various applications.
  
- **Model Architecture**:
  - **YOLO**: Uses a unified model for object detection, which processes the entire image in a single forward pass.
  - **Hugging Face**: Provides a range of models, including transformers for vision tasks, which can be fine-tuned for specific datasets.

- **Ease of Use**:
  - **YOLO**: Requires more effort to set up and train, but offers high performance for object detection tasks.
  - **Hugging Face**: Offers pre-trained models that are easy to fine-tune and integrate, with extensive support and documentation.

- **Application Speed**:
  - **YOLO**: Known for its speed and efficiency in real-time object detection.
  - **Hugging Face**: May not be as fast as YOLO for real-time detection, but excels in providing versatile and powerful pre-trained models for various tasks.

### Business Problem Case: Using images of interior rooms, and then identifying pictures, diplomas or other personally identifying objects.  Once identified, then the selected objects will be blurred.

**Using YOLO:**
1. **Dataset Preparation**: Annotate images with bounding boxes around pictures on the walls.
2. **Model Training**: Train the YOLO model with the annotated dataset.
3. **Inference**: Use the trained YOLO model to detect pictures on walls in new images.
