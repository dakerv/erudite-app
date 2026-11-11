# Deepfake Image Detection System

## Overview

The Deepfake Image Detection System is a web-based image classification application developed by Vanessa Daker.

The system uses **transfer learning with an EfficientNet-B0 convolutional neural network** to classify facial images into three categories:

* Real
* Face-Swapped
* Synthetic

The application combines a PyTorch-based deep learning model, a Flask backend, and a React + TypeScript frontend to provide an interactive interface for uploading an image and receiving a classification result with a confidence value.

The system is designed specifically for image-based deepfake detection and does not store uploaded photographs or require user accounts.

---

## Key Features

* Three-class deepfake image classification
* Real, synthetic, and face-swapped image detection
* EfficientNet-B0 transfer learning architecture
* ImageNet-pretrained model initialization
* Facial input validation
* Prediction confidence reporting
* Responsive web interface
* Flask REST API
* React + TypeScript frontend
* CPU-compatible inference
* No user authentication required
* Uploaded images are not permanently stored

---

# System Architecture

The application consists of three main components:

```text
┌─────────────────────────────┐
│       React Frontend        │
│      TypeScript + Vite      │
└──────────────┬──────────────┘
               │
               │ HTTP Request
               ▼
┌─────────────────────────────┐
│       Flask Backend         │
│       /predict endpoint     │
└──────────────┬──────────────┘
               │
               │ Image
               ▼
┌─────────────────────────────┐
│      EfficientNet-B0        │
│    PyTorch Classification   │
└──────────────┬──────────────┘
               │
               ▼
       ┌─────────────────┐
       │ Classification  │
       │ + Confidence    │
       └─────────────────┘
```

---

# How the System Works

The application follows this general workflow:

```text
Upload Image
     │
     ▼
Validate Image
     │
     ├── Invalid file ──────► Error message
     │
     ▼
Check for Visible Face
     │
     ├── No face detected ──► Reject image
     │
     ▼
Resize and Normalize
     │
     ▼
EfficientNet-B0
     │
     ▼
Three-Class Prediction
     │
     ▼
Calculate Class Probabilities
     │
     ▼
Display Result + Confidence
```

The model receives a facial image, processes it using the same image preprocessing required during model development, and produces probabilities for the three target classes.

---

# Classification Classes

The model predicts one of three classes:

| Class         | Description                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------- |
| Real          | An authentic facial image that is not generated or face-swapped                                |
| Synthetic     | A face generated artificially using a generative model                                         |
| Swapped       | An image in which a person's face has been replaced or transferred onto another person's image |

The frontend presents the classification result in user-oriented language such as:

* Likely Real
* Likely Swapped
* Likely Synthetic

along with the model's confidence value.


# Web Application

The web application provides an interface through which users can submit an image for analysis.

---

# Facial Input Validation

The application checks whether the uploaded image contains a detectable face before performing classification.

Images in which no face is detected are rejected because the model was developed specifically for facial-image classification.

The system therefore distinguishes between:

* Invalid image files
* Images without a detected face
* Valid facial images suitable for classification

Face detection is performed using OpenCV's Haar Cascade classifier within the backend.

---

# Technologies

## Machine Learning

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Scikit-learn

## Backend

* Flask
* PyTorch
* Pillow
* OpenCV
* NumPy

## Frontend

* React
* TypeScript
* Vite
* Tailwind CSS

---

### Development Hardware

* **Computer:** HP EliteBook 840 G6
* **Processor:** Intel Core i5-8265U
* **Memory:** 8 GB RAM
* **Graphics:** Intel UHD Graphics 620
* **Training device:** CPU

The CPU-only environment influenced training time and contributed to the use of a relatively lightweight architecture and small batch size.

---

# Limitations

Despite the high test-set classification performance, the system has several limitations.

### Image-Only Detection

The current system is designed for still images and does not analyse video, audio, or temporal inconsistencies.

### Dataset Dependence

Performance depends on the characteristics of the datasets used during development. A model may encounter different manipulation techniques, image-generation methods, compression levels, or image distributions in real-world applications.

### Synthetic Image Diversity

The synthetic class is based primarily on StyleGAN-generated faces. Other generative models may produce different visual characteristics.

### Face-Swapped Data

The face-swapped class is based on a specific source dataset. Different face-swapping techniques may introduce artifacts that differ from those represented in the training data.

### Computational Constraints

Training was performed on a CPU-based laptop with 8 GB of RAM. Access to more powerful computational resources would make it practical to investigate larger datasets and additional architectures.

### Confidence Interpretation

The confidence value represents the model's output probability and should not be interpreted as absolute certainty.

### Face Detection Dependency

The application relies on facial detection before classification. A valid facial image may occasionally fail the detection step because of factors such as pose, image quality, occlusion, or lighting.

---

# Future Development

Potential areas for future development include:

* Expanding the diversity and size of the training dataset
* Including synthetic images from additional generative architectures
* Including additional face-swapping methods
* Evaluating other CNN architectures
* Investigating transformer-based image classification models
* Testing the system against previously unseen deepfake generation techniques
* Improving facial-region detection and preprocessing
* Adding explainable-AI techniques such as saliency maps
* Extending the system to video-based deepfake detection
* Improving deployment efficiency
* Evaluating performance across different image compression levels and resolutions

---


# Acknowledgements

The project makes use of publicly distributed datasets and open-source machine-learning and software-development technologies.

The dataset sources, software libraries, and frameworks used in the project are acknowledged for supporting the development and evaluation of the system.

---

# Author

**Vanessa Elinam Daker**

Software Engineer
