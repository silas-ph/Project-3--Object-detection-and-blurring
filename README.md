# Project 3: Video Processing with YOLO and Streamlit for Privacy-Preserving Object Detection

## Group 2 Team Members:
- Christoph Guenther
- Keegan Nohavec
- Silas Phillips

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [GitHub Organization](#github-organization)
4. [Technology Stack](#technology-stack)
5. [Step 1 – Creating a Custom Dataset with Roboflow](#step-1--creating-a-custom-dataset-with-roboflow)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Image Annotation](#image-annotation)
    - [Split into Train, Validation, and Test Sub-Datasets](#split-into-train-validation-and-test-sub-datasets)
    - [Image Augmentation](#image-augmentation)
6. [Step 2 – YOLO Model Training](#step-2--yolo-model-training)
    - [Advantages of YOLO](#advantages-of-yolo)
    - [Model Training](#model-training)
    - [Object Detection Metrics Used](#object-detection-metrics-used)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
        - [Epochs](#epochs)
        - [Batch Size](#batch-size)
        - [Patience](#patience)
        - [IoU Threshold](#iou-threshold)
    - [Summary of Results](#summary-of-results)
7. [Step 3 – Streamlit Application for Video Processing](#step-3--streamlit-application-for-video-processing)
    - [Overview of the Video Processing Solution](#overview-of-the-video-processing-solution)
    - [Privacy Preservation with Gaussian Blur](#privacy-preservation-with-gaussian-blur)
8. [System Evaluation](#system-evaluation)
9. [Challenges and Solutions](#challenges-and-solutions)
10. [Future Improvements](#future-improvements)
    - [Expanded Images for Training](#expanded-images-for-training)
11. [Conclusion](#conclusion)
12. [Resources](#resources)
13. [References](#references)

## Executive Summary
We trained a YOLO (You Look Only Once) AI model[^1,^2,^3,^4,^5,^6] to successfully detect picture objects with bounding boxes in images. We applied the YOLO model to videos to detect these objects and blurred them using a Gaussian Blur[^7] algorithm. We built a web application using Streamlit[^8] allowing users to upload their own videos for processing and download the processed video.

This application is intended for users who want to de-identify objects in videos before publishing them to a (semi-)public audience.

## Introduction
This report presents a comprehensive overview of implementing a privacy-preserving object detection solution using YOLO (You Only Look Once) for model training and inference integrated with a Streamlit application for video processing. The system detects objects in video streams and applies Gaussian blur to specific regions ensuring privacy and security in the processed output. This project combines state-of-the-art object detection techniques with practical video processing showcasing an end-to-end solution for a real-world application.

## GitHub Organization
This GitHub repository contains the following subfolders:
- `Docs` – Contains the presentation slide deck describing the project.
- `Images` – Contains the images used in our final model organized into “train” “valid” and “test” subfolders.
- `Jupyter Notebooks` – Contains two Jupyter notebooks:
  - `Project_3_model_training.ipynb` – Contains the code to train a YOLO model.
  - `RetrieveYOLOMetrics.ipynb` – Contains the code to retrieve the metrics for each valid training run.
- `Runs` – Contains a subfolder with saved data and metrics for each training run using different YOLO models and hyperparameters.
- `Streamlit` – Contains `streamlit.py` the python script for the end user web application to process videos and `best.pt` a file generated by YOLO containing a set of its parameters.

## Technology Stack
Our application uses several key technologies:
- **Data Preparation**: Utilizing Roboflow[^9] and Kaggle[^10] for dataset creation and annotation.
- **Dataset Management**: Utilizing Roboflow to create a custom dataset.
- **Model Training**: Employing YOLO for object detection training it on our custom dataset.
- **Video Processing**: Implementing a Streamlit application for video input and processing.
- **Privacy Preservation**: Applying Gaussian blur to detected objects for privacy.

## Step 1 – Creating a Custom Dataset with Roboflow
We used Roboflow to create custom image datasets that can be used directly to train our YOLO models. In particular, we were able to take advantage of the following Roboflow features:

### Collaboration and Quality Control
Roboflow allows multiple team members to work on the same annotation project simultaneously. Since annotation is performed through a user interface, it guarantees the consistency of the label files containing the annotation information.

### Export Flexibility
The label files can be exported in various formats, including the ones that different YOLO versions require.

### API Integration
When creating a custom dataset, Roboflow provides a code snippet that can be directly copied into the code that is used to train the YOLO models. This code snippet will create the folder structure that YOLO expects with images in `train`, `val`, and `test` folders.

Creating a custom image dataset for use to train YOLO models requires these tasks:
- Exploratory Data Analysis (EDA)
- Image Annotation
- Split into Train, Validation, and Test sub-datasets
- Image Augmentation

### Exploratory Data Analysis (EDA)
Using a single Kaggle dataset simplified our EDA. All source images were already consistently formatted as JPG images and uniformly sized to 224x224 pixels. Nevertheless, we used Roboflow’s preprocessing functions to ensure all images were in fact of size 224x224 pixels and auto-oriented. Using Roboflow, we further simplified EDA by assigning all images to a single class called “Pictures-on-Wall.” We used 3415 images from the Kaggle image set in our final dataset.

### Image Annotation
Image annotation consists of identifying class objects in the image, assigning it to the correct class, and identifying bounding boxes around each object, specifying the location of each object in the image.

Roboflow's interface enabled efficient image annotation. We used these features:
- Drag-and-Drop Bounding Boxes: Easily created around objects of interest
- Keyboard Shortcuts: Accelerated the annotation process
- Smart Polygon Tool: Used to manually select pictures on walls or shelves

We annotated all 3415 images in the dataset using these tools.


This image shows the interface of the image before annotation.

![Annotated Image](Images/Robo%201.png)

This image shows the image with bounding boxes after annotation.

![Dataset Split](Images/Robo%202.png)

### Split into Train, Validation, and Test Sub-Datasets
YOLO expects datasets to be split into `train`, `val`(idation), and `test` sub-datasets. For YOLO to be able to process these sub-datasets, they have to be placed in separate folders, with the path to each subfolder provided in a `data.yaml` file. In addition, YOLO requires the label file for each image file to be of a specific format and placed in a specific path.  
We used Roboflow's "Create Dataset" feature to accomplish these tasks. It creates the folder structure and all required files (such as the `data.yml` file, the label files, etc.) expected by YOLO. The only thing we had to do was edit the paths to the `train`, `val`, and `test` folders to make sure they were what YOLO expects.



The following screenshot shows the Roboflow dataset with train, validation, and test split.


![Augmented Images](Images/Robo%203.png)

### Image Augmentation
To efficiently increase the number of training images to build the YOLO model, we used Roboflow's augmentation capabilities. We augmented our initial training images dataset using the following augmentations:
- Crop: Random zooming from 0% to 20%.
- Rotation: Random rotations between -15° to +15°.
- Brightness: To simulate different lighting conditions, brightness was randomly adjusted from -15% to +15%.
- Blur: Up to 2.5 pixels.


In summary, by leveraging Roboflow's comprehensive features, we were able to create a high-quality, well-annotated dataset, crucial for training our YOLO model effectively.

## Step 2 – YOLO Model Training

### Advantages of YOLO
- **Speed**: Single-stage detection allows for detection of object classes present and identification of bounding boxes in one pass through the image.
- **Accuracy**: YOLO architecture and advanced loss functions lead to precise localization of objects and enhance multi-scale object detection.
- **Flexibility**: Easily adjustable model sizes from nano to extra-large with different numbers of parameters.
- **Transfer of Best Parameters**: For each run, YOLO provides a file called `best.pt` containing the best parameters it found for that run. This file and hence these parameters can then be used as the starting point for subsequent runs.
- **Anchor-free Detection (YOLO version 4 and higher)**: Simplifies the detection process and often leads to better performance on small objects.

### Model Training
We used Google CoLaboratory (CoLab) for all of our training runs. We tested various runtime versions using the CPU and various GPUs. We achieved the best performance using an Nvidia A100 GPU. We used multiple processors for our last 10 training runs. This improved our processing speed by about 60%. To further optimize processing speed, we also used image caching, which allows the GPU to cache training and validation images (depending on its resources). We achieved this by setting the YOLO hyperparameter called `Cache` to `True`.

In total, we trained our model 17 times using different YOLO versions, models, and hyperparameters. We started the training process using YOLO version 8[^11], using the Medium and Extra-large models with 1487 images. To improve the accuracy of our model, we increased the dataset to 3415 images and ran 2 more iterations using YOLO version 8, noting improved precision.

Encouraged by the improved precision due to increasing the number of images in our dataset, we used Roboflow’s augmentation tool to further increase the image count to 8145 images. Time constraints prevented us from annotating more images, so we used augmentation instead. We performed 4 more training runs with YOLO version 8 on all 8145 images.

After further research, we discovered that YOLO version 10 promised to be faster without sacrificing precision[^12]. Therefore, we decided to switch to YOLO version 10 and ran 5 iterations using the medium model.

We did not preserve any performance metrics of our initial 8 training runs. These runs were used to improve the size of our datasets, optimize the Google Colab setup, and narrow down the YOLO model to use.

Training logs of each of the runs used are as follows:

![Model Metrics](Images/Training_logs%204.png)

### Object Detection Metrics Used
We used the following metrics to evaluate each of our training runs:
- **P (Precision)**: Precision quantifies the proportion of true positives among all positive predictions, assessing the model's capability to avoid false positives. 
- **R (Recall)**: Recall calculates the proportion of true positives among all actual positives, measuring the model's ability to detect all instances of a class.
- **mAP50**: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. It's a measure of the model's accuracy, considering only the "easy" detections.
- **mAP50-95**: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95. It provides a comprehensive view of the model's performance across different levels of detection difficulty.

Here IoU refers to the ratio of the intersection between a predicted bounding box and the real bounding box (ground truth bounding box) over the union of the predicted bounding box and the real bounding box.

We chose these particular metrics because YOLO provides them after each epoch.

The following table shows these metrics for the runs for which we preserved the best parameters of the run:

![Training Interface](Images/Param%20runs%205.png)

### Hyperparameter Tuning
We adjusted the epoch, batch size, patience, and IoU (Intersection over Union) hyperparameters. To view the hyperparameters of the model we chose for video processing, please reference [args.yaml](https://github.com/silas-ph/Project-3--Object-detection-and-blurring/blob/main/Runs/V10M_8195im_600ep_05IoU_1cls/args.yaml).

Here's a detailed look at our hyperparameter tuning process:

#### Epochs
- **Definition**: The number of complete passes through the entire training dataset.
- **Tuning Process**:
  - **Initial Setting**: We started with the default of 100 epochs.
  - **Experimentation**: We tried values ranging from 10 to 900 epochs.
  - **Observation**: Increasing epochs generally improved performance up to a point, after which we saw diminishing returns.
  - **Final Setting**: We settled on 600 epochs, which provided a good balance between performance and training time.
  - **Impact**: More epochs allowed the model to learn more complex features, improving detection accuracy, especially for challenging objects.

#### Batch Size
- **Definition**: The number of training samples processed before the model is updated.
- **Tuning Process**:
  - **Initial Setting**: We began with the default batch size of 16.
  - **Consideration**: Larger batch sizes require more memory but can lead to more stable gradients.
  - **Final Setting**: We chose a batch size of -1 (auto), which offered a good trade-off between memory usage and training stability.
  - **Impact**: The auto batch size allowed for more stable gradient updates and slightly faster training times without sacrificing model performance.

#### Patience
- **Definition**: The number of epochs to wait before early stopping if no improvement is seen.
- **Tuning Process**:
  - **Initial Setting**: The default patience was set to 100 epochs.
  - **Experimentation**: We tried values ranging from 20 to 100.
  - **Observation**: Lower patience values risked stopping training too early, while higher values could lead to unnecessary computation.
  - **Final Setting**: We settled on a patience of 100 epochs.
  - **Impact**: This setting helped us strike a balance between giving the model enough time to improve and avoiding wasted computation on plateaued performance.

#### IoU Threshold
- **Definition**: IoU is a measure that quantifies the overlap between a predicted bounding box and a ground truth bounding box. It plays a fundamental role in evaluating the accuracy of object localization.
- **Tuning Process**:
  - **Initial Setting**: The default IoU threshold was 0.7.
  - **Experimentation**: We tested values from 0.4 to 0.7.
  - **Consideration**: Lower thresholds increase recall but may lead to more false positives, while higher thresholds increase precision but may miss detections.
  - **Final Setting**: We chose an IoU threshold of 0.5.
  - **Impact**: This slightly higher IoU threshold improved the precision of our detections, which was crucial for our privacy-preserving application to ensure accurate blurring.

### Summary of Results
After our hyperparameter tuning and training process, we chose the following configuration to use in the subsequent video processing:
- **Model**: yolov10m - 16,451,542 parameters.
- **Hyperparameters**:
  - Epochs = 600
  - Cache = True
  - Batch Size = -1
  - Patience = 100
  - IoU = 0.5

We chose this configuration because it provided the highest precision, with the other metrics being comparable to the other configurations we tried. In particular, this model yielded the following metrics:
- Precision = 0.792
- Recall = 0.552
- mAP50 = 0.637
- mAP50-95 = 0.425

## Step 3 – Streamlit Application for Video Processing

### Overview of the Video Processing Solution
This solution allows users to upload a video, detect objects within it, and blur those objects to ensure privacy. It integrates object detection using a YOLO model with a web application built using Streamlit.

The process begins by importing necessary tools to handle tasks such as creating the web interface, processing the video, and handling numerical operations. The YOLO model, which is trained to detect specific objects, is loaded and ready for use.

To ensure privacy, the solution includes a function that applies a Gaussian blur to specific areas in the video frames. This involves identifying the regions (bounding boxes) where objects are detected, applying a blur to those areas, and then reinserting the blurred sections back into the video frame. This ensures that sensitive parts of the video are obscured.

The solution also initializes a tracking system for these objects. This system uses trackers to follow the objects across multiple frames, ensuring they remain blurred even as they move. The trackers are updated regularly to maintain accuracy and handle any new objects that appear.

The main part of the solution processes the entire video. It reads the video frame by frame, detects objects every few frames to keep track of new and moving objects, and applies the blur to these detected areas. This processing loop continues until the entire video has been handled, and the progress is shown to the user through a progress bar.

The web application allows users to upload their videos, process them to detect and blur objects, and then download the processed videos. The application interface is simple and user-friendly, guiding users through uploading a video, waiting for the processing to complete, and downloading the finished product.

The multi-tracker system is a key feature of this solution. By initializing and updating multiple trackers, the system can handle multiple objects in the video simultaneously, ensuring that all detected objects remain blurred throughout the video. This approach also adapts to new objects entering the frame or existing objects moving, maintaining robust and accurate tracking.

The Gaussian blur applied to the objects uses a specific method that provides a strong blur effect while keeping the overall video context clear. This balance ensures that the video remains useful for viewing while protecting the privacy of any sensitive content.

Overall, this solution seamlessly combines advanced object detection with practical video processing in a user-friendly web interface, making it accessible and efficient for users who need to ensure privacy in their video content.

### Privacy Preservation with Gaussian Blur

#### Implementing Gaussian Blur
The Gaussian blur is applied with a kernel size of 51x51, which provides a strong blurring effect while maintaining the overall structure of the image.

#### Balancing Privacy and Usability
The choice of blur intensity (controlled by the kernel size and sigma value) is crucial:
- Too little blur may not provide adequate privacy protection.
- Too much blur can make the video unusable for certain applications.

We conducted experiments with different blur levels to find an optimal balance between privacy preservation and video usability.

## System Evaluation
We evaluated our system on several criteria:
- **Detection Accuracy**: Measured the model's ability to correctly identify objects in video frames.
- **Processing Speed**: Assessed the system's ability to process video in real-time or near-real-time.
- **Privacy Effectiveness**: Evaluated how well the Gaussian blur protected the privacy of detected objects.
- **User Experience**: Gathered feedback on the Streamlit app's usability and performance.

## Challenges and Solutions
Throughout the project, we encountered several challenges:
- **Real-time Processing**: Achieving real-time performance was challenging with high-resolution videos. 
  - **Solution**: We implemented frame skipping and reduced resolution for faster processing. However, we never achieved real-time processing. More work is required in this area.
- **False Positives**: The model sometimes detected objects incorrectly, leading to unnecessary blurring. 
  - **Solution**: We fine-tuned the model on more diverse data and adjusted the confidence threshold. However, we still encounter a higher number of false positives than we would like. More systematic model development is required to minimize the number of false positives. 
- **Temporal Consistency**: Blurred regions could flicker or change size between frames. 
  - **Solution**: We implemented a simple tracking algorithm to smooth the bounding boxes across frames.
- **Memory Management**: Processing large videos led to memory issues in Streamlit. 
  - **Solution**: We implemented batch processing of video chunks to manage memory more efficiently.

## Future Improvements
Looking ahead, there are several avenues for improving our system:
- **Advanced Tracking**: Implement more sophisticated object tracking algorithms for smoother blur application across frames.
- **Selective Blurring**: Develop a system to selectively blur only certain classes of objects based on user preferences.
- **Edge Deployment**: Optimize the model and application for deployment on edge devices for local video processing.
- **Interactive Blur Control**: Add UI elements in Streamlit to enable users to adjust blur intensity in real-time.

### Expanded Images for Training
- **Diverse Source Utilization**: 
  - Expand beyond Kaggle to include other datasets like Roboflow, COCO, Open Images, and domain-specific repositories.
  - Collaborate with industry partners to obtain real-world application-specific data.
- **Synthetic Data Generation**: 
  - Utilize 3D modeling software to generate synthetic images with perfect annotations.
  - Employ Generative Adversarial Networks (GANs) to create realistic, diverse images.
- **Data Mining**: 
  - Implement web scraping tools to collect relevant images from the internet.
  - Use APIs of stock photo websites to access a wide range of high-quality images.

## Conclusion
This project demonstrated the successful integration of YOLO object detection with a Streamlit application for privacy-preserving video processing. By combining state-of-the-art deep learning techniques with practical video processing and a user-friendly interface, we created a system that effectively detects objects in video streams and applies privacy-preserving blur.

The use of Roboflow and Kaggle for dataset creation, along with YOLO for object detection, provided a solid foundation for accurate and efficient object detection. The Streamlit application offered an accessible and interactive platform for video processing, making the technology easily usable for end-users.

This project not only resulted in a functional privacy-preserving video processing system but also provided valuable insights into the challenges and considerations of implementing AI-driven video analysis tools. As privacy concerns continue to grow in importance, systems like this will play a crucial role in balancing the benefits of video analytics with the need for privacy protection.

## Resources  
Medium.com, ChatGpt, Claude

## References
[^1]: Redmon Joseph et al. (2016 May 9). *You Only Look Once: Unified Real-Time Object Detection* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2016 https://arxiv.org/abs/1506.02640.
[^2]: Redmon Joseph and Ali Farhadi (2016 December 25). *YOLO9000: Better Faster Stronger* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017 https://arxiv.org/abs/1612.08242.
[^3]: Redmon Joseph and Ali Farhadi (2018 Apr 8). *YOLOv3: An Incremental Improvement* arXiv preprint 2018 https://arxiv.org/abs/1804.02767.
[^4]: Bochkovskiy Alexey et al. (2020 Apr 23). *YOLOv4: Optimal Speed and Accuracy of Object Detection* arXiv preprint 2020 https://arxiv.org/abs/2004.10934.
[^5]: Ultralytics (2024) *YOLOv5* Ultralytics Inc. https://docs.ultralytics.com/models/yolov5/ as accessed on 8/1/2024.
[^6]: Rajput Vishal (2024 May 26). *YOLOv10: Object Detection King Is Back* Medium.com https://medium.com/aiguys/yolov10-object-detection-king-is-back-739eaaab134d accessed on 8/1/2024.  
[^7]: *Gaussian blur* Wikipedia https://en.wikipedia.org/wiki/Gaussian_blur accessed on 8/1/2024.  
[^8]: *Streamlit • A faster way to build and share data apps* Snowflake Inc. https://streamlit.io/ accessed on 8/1/2024. 
[^9]: Reni Robin (2020). *House Rooms Image Dataset* Kaggle https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset/data accessed on 8/1/2024. License: CC0 1.0 UNIVERSAL Deed.  
[^10]: *Roboflow: Computer vision tools for developers and enterprises* Roboflow.com https://roboflow.com/ accessed on 8/1/2024.
[^11]: Ultralytics (2024) *YOLOv8* Ultralytics Inc. https://docs.ultralytics.com/models/yolov8/ as accessed on 8/1/2024.
[^12]: Ultralytics (2024) *YOLOv10* Ultralytics Inc. https://docs.ultralytics.com/models/yolov10/ as accessed on 8/1/2024.

YOLO: AGPL-3.0 License: This OSI-approved open-source license is ideal for students and enthusiasts promoting open collaboration and knowledge sharing. 
@article{THU-MIGyolov10
  title={YOLOv10: Real-Time End-to-End Object Detection}
  author={Ao Wang Hui Chen Lihao Liu et al.}
  journal={arXiv preprint arXiv:2405.14458}
  year={2024}
  institution={Tsinghua University}
  license = {AGPL-3.0}
}
