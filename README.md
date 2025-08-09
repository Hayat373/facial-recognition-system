# Facial Recognition System

A web-based facial recognition system built with Python, OpenCV, TensorFlow, and Flask. Designed for applications like security or social media tagging, this project enables users to capture face images, train a convolutional neural network (CNN) to recognize faces, and identify individuals in real-time via a webcam through a Flask-based web interface.

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Capture Face Images](#capture-face-images)
  - [Train the Model](#train-the-model)
  - [Run the Web Application](#run-the-web-application)
  - [Recognize Faces Directly](#recognize-faces-directly)


---

## Features
- **Face Capture**: Collect face images using a webcam and store them for training.
- **Model Training**: Train a CNN model to recognize faces based on captured images.
- **Real-Time Recognition**: Identify faces in real-time using a webcam, supporting multiple individuals.


---

## Prerequisites
- **Python**: Version 3.8 or higher
- **Webcam**: Required for capturing and recognizing faces
- **Git**: To clone the repository
- **Internet Connection**: For installing dependencies

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hayat373/faceRecogistionAi.git
   cd faceRecogistionAi
   ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

 3. **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
    ```

## Usage

### Capture Face Images

 ```bash 
  python utils/capture_faces.py
```
### Train the Model
   ```bash
   python utils/train_model.py
   ```

### Recognize Faces Directly

   ```bash 
   python utils/recognize_faces.py
   ```
