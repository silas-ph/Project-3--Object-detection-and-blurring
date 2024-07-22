### Elevator Pitch

**Protecting Privacy in Insurance Claims with Advanced AI**

Imagine a world where your privacy is seamlessly integrated into every aspect of your life, especially in shared spaces. Our groundbreaking AI object detection technology makes this vision a reality by ensuring that your personal moments stay private, even in public settings.

Our cutting-edge solution leverages advanced AI algorithms to identify and blur pictures on walls from uploaded videos. This intuitive user interface allows for effortless video uploads, and our intelligent technology detects and blurs sensitive images, safeguarding your privacy and that of others.

No longer worry about inadvertently sharing sensitive information or personal photos. Our system is designed to be fast, efficient, and unobtrusive, giving you peace of mind without disrupting your daily activities. Embrace a future where privacy is prioritized and effortlessly maintained with our AI-powered image blurring technology.

Join us in redefining privacy protection for insurance claims and beyond.




### Business Problem: Videos captured by insureds will have sensitive/valuable and personally identifiable information.
### Solution: Identify objects on walls and RX Bottles and then blur them.

### Image dataset resources
-**[https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset](https://universe.roboflow.com/ibee/house-tfo7u/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

-**[https://images.cv/dataset/pill-bottle-image-classification-dataset](https://images.cv/dataset/pill-bottle-image-classification-dataset)

**Minimum Viable Product (MVP):**
Our initial focus is to develop a model using images only to do the following:

1. **Geometric Shape Detection:**  Accurately identify geometric shapes (e.g., rectangles, squares, circles) present on walls in images. These shapes could represent pictures, diplomas, artwork, or other wall-mounted objects.
2. **Selective Blurring:** Apply blurring techniques to obscure the identified geometric shapes, effectively anonymizing the content within them.
3. **Pill Bottle Detection & Blurring:** Precisely identify pill bottles within images and apply blurring to protect sensitive information.
4. **Public Dataset Training:** Train the model using publicly available image datasets to ensure robustness and adaptability.

**Stretch Goals:**
Once the MVP is successfully implemented, we aim to expand the system's capabilities to:

1. **Video Processing:** Extend the functionality to process video files, breaking them down into individual frames.
2. **Blurring on Frames:** Apply the shape detection and blurring algorithms to each frame of the video.
3. **3D Model Integration:** Combine the blurred image frames with 3D models of the corresponding rooms, creating a comprehensive anonymized representation. 


### Accelerated Project Outline for Image Classification and Object Blurring (4 Weeks)

#### Phase 1: Project Planning and Setup (0.5 Week)
- **Task 1.1**: Define Project Scope and Objectives
  - **Description**: Clarify the goals, deliverables, and success criteria of the project.
  - **Time Frame**: 2 days
- **Task 1.2**: Assemble Project Team
  - **Description**: Identify team members and assign roles and responsibilities.
  - **Time Frame**: 1 day
- **Task 1.3**: Set Up Project Infrastructure
  - **Description**: Establish the development environment, tools, and libraries.
  - **Time Frame**: 2 days

#### Phase 2: Data Acquisition and Preprocessing (1 Week)
- **Task 2.1**: Extract Frames from Video
  - **Description**: Develop a script to break down the video into individual frames.
  - **Time Frame**: 1 day
- **Task 2.2**: Data Collection and Annotation
  - **Description**: Use a pre-labeled dataset if available. Otherwise, manually label a small set of images using LabelImg.
  - **Time Frame**: 3 days
- **Task 2.3**: Data Augmentation
  - **Description**: Augment the dataset to improve model robustness (e.g., rotation, scaling, flipping).
  - **Time Frame**: 1 day

#### Phase 3: Model Development (1.5 Weeks)
- **Task 3.1**: Develop Picture and RX Pill Detection Model
  - **Description**: Use a pre-trained model like YOLOv5 or Faster R-CNN and fine-tune it on the annotated data.
  - **Time Frame**: 1 week
- **Task 3.2**: Develop Object Detection Model for RX Bottles
  - **Description**: Use a pre-trained model and fine-tune it on the annotated data.
  - **Time Frame**: 1 week (overlap with Task 3.1)

#### Phase 4: Model Evaluation and Optimization (1 Week)
- **Task 4.1**: Evaluate Models
  - **Description**: Evaluate the performance of the detection models using metrics like accuracy, precision, recall, F1-score, and mean Average Precision (mAP).
  - **Time Frame**: 3 days
- **Task 4.2**: Optimize Models
  - **Description**: Optimize model parameters and retrain if necessary to improve performance.
  - **Time Frame**: 2 days

#### Phase 5: Image Processing (0.5 Week)
- **Task 5.1**: Implement Blurring of Detected Pictures
  - **Description**: Develop a script to blur the detected pictures on walls in the images.
  - **Time Frame**: 2 days
- **Task 5.2**: Implement Blurring of Objects on RX Pill bottles
  - **Description**: Develop a script to blur the detected RX Bottles objects in the images.
  - **Time Frame**: 2 days


### Summary
- **Total Duration**: 4 weeks

### Detailed Tasks Breakdown:

#### Week 1: Planning and Data Preparation
- **Day 1-2**: Define project scope, assemble team, and set up infrastructure.
- **Day 3-6**: Collect and annotate data using LabelImg.
- **Day 7**: Perform data augmentation.

#### Week 2: Model Development
- **Day 8-14**: Fine-tune pre-trained models for picture and RX bottle detection.
- **Day 8-14**: Fine-tune pre-trained models for object detection on countertops (overlap).

#### Week 3: Evaluation and Optimization
- **Day 15-17**: Evaluate model performance.
- **Day 18-19**: Optimize model parameters.

#### Week 4: Image Processing and Integration
- **Day 20-21**: Implement blurring of detected pictures.
- **Day 22-23**: Implement blurring of objects on countertops.
- **Day 24-28**: Integrate all components and conduct testing.


