# Deepfake Image Detection System

## Overview

The **Deepfake Image Detection System** is a web-based image classification application developed as a final-year project at the **University of Ghana, Legon**.

The system uses **transfer learning with an EfficientNet-B0 convolutional neural network** to classify facial images into three categories:

* **Real**
* **Face-Swapped**
* **Synthetic**

The application combines a **PyTorch-based deep learning model**, a **Flask backend**, and a **React + TypeScript frontend** to provide an interactive interface for uploading an image and receiving a classification result with a confidence value.

The system is designed specifically for **image-based deepfake detection** and does not store uploaded photographs or require user accounts.

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
| **Real**      | An authentic facial image that is not generated or face-swapped                                |
| **Synthetic** | A face generated artificially using a generative model                                         |
| **Swapped**   | An image in which a person's face has been replaced or transferred onto another person's image |

The frontend presents the classification result in user-oriented language such as:

* **Likely Real**
* **Likely Swapped**
* **Likely Synthetic**

along with the model's confidence value.

---

# Dataset

The final training dataset contains **18,000 images**, evenly distributed across the three classes.

| Class        | Number of Images |
| ------------ | ---------------: |
| Real         |            6,000 |
| Synthetic    |            6,000 |
| Face-Swapped |            6,000 |
| **Total**    |       **18,000** |

## Real Images

The real-image class was constructed from two sources:

* **FFHQ (Flickr-Faces-HQ)**
* **Celeb-DF real images**

The final real-image dataset contains:

* 3,000 FFHQ images
* 3,000 CelebDF real images

for a total of **6,000 real images**.

## Synthetic Images

The synthetic class consists of artificially generated facial images produced using **StyleGAN3**.

The source is referred to in the project as the **StyleGAN3 Fake Faces / StyleGAN3 Synthetic Face Image Dataset**.

A total of **6,000 synthetic images** were selected for the final dataset.

## Face-Swapped Images

The swapped class was obtained from the dataset titled:

**DeepFake(face swapped) images using FFHQ dataset**

The dataset contains face-swapped images generated through facial manipulation techniques.

A total of **6,000 images** were selected for the final dataset.

---

# Data Preprocessing

Before training, the images were processed to provide a consistent input format.

The preprocessing pipeline includes:

1. Face detection
2. Identification of the largest detected face
3. Facial-region cropping
4. Addition of a facial-region margin
5. Resizing to **224 × 224 pixels**
6. Dataset splitting
7. Image normalization

The system uses **MTCNN** when available for face detection and provides a Haar Cascade-based fallback.

The final dataset was divided into training, validation, and testing subsets.

| Class     |   Training | Validation |   Testing |      Total |
| --------- | ---------: | ---------: | --------: | ---------: |
| Real      |      4,200 |        900 |       900 |      6,000 |
| Synthetic |      4,200 |        900 |       900 |      6,000 |
| Swapped   |      4,200 |        900 |       900 |      6,000 |
| **Total** | **12,600** |  **2,700** | **2,700** | **18,000** |

The test set was kept separate from training and model selection and was used only for the final evaluation.

---

# Model

## EfficientNet-B0

The classification model is based on **EfficientNet-B0**, a convolutional neural network architecture designed to achieve a balance between model accuracy and computational efficiency.

Transfer learning was used rather than training the network entirely from scratch.

The model was initialized using **ImageNet-pretrained weights**, after which the original 1,000-class ImageNet classifier was replaced with a three-class classifier.

```text
EfficientNet-B0
      │
      ▼
Feature Extraction
      │
      ▼
1280-dimensional representation
      │
      ▼
Linear Layer
1280 → 3
      │
      ▼
Real / Synthetic / Swapped
```

---

# Training Configuration

The final model was trained using the following configuration:

| Parameter         | Value             |
| ----------------- | ----------------- |
| Architecture      | EfficientNet-B0   |
| Learning approach | Transfer Learning |
| Initial weights   | ImageNet          |
| Number of classes | 3                 |
| Batch size        | 8                 |
| Optimizer         | Adam              |
| Learning rate     | 0.0001            |
| Loss function     | CrossEntropyLoss  |
| Image size        | 224 × 224         |
| Device            | CPU               |
| Total epochs      | 15                |

The final model was selected using **validation accuracy**.

The highest validation accuracy was obtained at **Epoch 13**.

---

# Final Model Results

The final selected model was the **Epoch 13 checkpoint**.

### Validation Performance

**Validation Accuracy: 98.59%**

The Epoch 13 model was subsequently evaluated on the completely held-out test set.

### Final Test Performance

**Test Accuracy: 98.37%**

The model correctly classified:

**2,656 out of 2,700 test images.**

```text
Correct predictions: 2656 / 2700
Test Accuracy:       98.37%
```

---

# Confusion Matrix

The final test confusion matrix is shown below.

Class order:

```text
Real
Synthetic
Swapped
```

| Actual / Predicted | Real | Synthetic | Swapped |
| ------------------ | ---: | --------: | ------: |
| **Real**           |  882 |        18 |       0 |
| **Synthetic**      |   24 |       876 |       0 |
| **Swapped**        |    2 |         0 |     898 |

The matrix shows that the majority of classification errors occurred between the **real** and **synthetic** classes.

---

# Classification Report

| Class                |  Precision |     Recall |   F1-Score |   Support |
| -------------------- | ---------: | ---------: | ---------: | --------: |
| Real                 |     97.14% |     98.00% |     97.57% |       900 |
| Synthetic            |     97.99% |     97.33% |     97.66% |       900 |
| Swapped              |    100.00% |     99.78% |     99.89% |       900 |
| **Macro Average**    | **98.37%** | **98.37%** | **98.37%** | **2,700** |
| **Weighted Average** | **98.37%** | **98.37%** | **98.37%** | **2,700** |

Because each class contains the same number of test images, the macro and weighted averages are identical in this evaluation.

---

# Training Development

An earlier development experiment used a smaller dataset containing **9,000 images**, with 3,000 images per class.

That experiment achieved a test accuracy of:

**96.07%**

The final experiment expanded the dataset to **18,000 images**, with 6,000 images per class, and achieved:

**98.37% test accuracy.**

| Experiment   | Dataset Size | Test Accuracy |
| ------------ | -----------: | ------------: |
| Experiment 1 |        9,000 |        96.07% |
| Experiment 2 |       18,000 |    **98.37%** |

The experiments represent stages in the development and evaluation of the system. The final deployed model is based on **Experiment 2**.

---

# Web Application

The web application provides an interface through which users can submit an image for analysis.

The general interaction is:

```text
User selects image
        │
        ▼
Frontend validates input
        │
        ▼
Image sent to Flask API
        │
        ▼
Backend checks image
        │
        ▼
Face detection
        │
        ▼
EfficientNet-B0 inference
        │
        ▼
Prediction probabilities
        │
        ▼
Result returned to frontend
        │
        ▼
Classification + confidence displayed
```

---

# Facial Input Validation

The application checks whether the uploaded image contains a detectable face before performing classification.

Images in which no face is detected are rejected because the model was developed specifically for facial-image classification.

The system therefore distinguishes between:

* Invalid image files
* Images without a detected face
* Valid facial images suitable for classification

Face detection is performed using **OpenCV's Haar Cascade classifier** within the backend.

---

# Prediction Confidence

The model produces a probability distribution over the three classes.

For example:

```text
Real:       0.03
Synthetic:  0.95
Swapped:    0.02
```

The class with the highest probability is returned as the predicted class.

The application displays this probability as the prediction confidence.

The confidence value represents the model's output probability for the selected class. It should not be interpreted as a guarantee that the classification is correct.

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
* React Router
* Lucide
* Framer Motion
* Radix UI components

---

# Project Structure

A simplified version of the project structure is:

```text
deep_fake_is-this-real/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── models (experiment two)/
│   └── efficientnet_b0.pth
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── raw_data/
│   ├── real/
│   ├── synthetic/
│   └── swapped/
│
├── cropped_faces/
│
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── test_model2.py
│
└── README.md
```

The exact directory structure may vary depending on the development environment and which datasets or intermediate files are retained locally.

---

# Installation

## Requirements

The project requires:

* Python 3.x
* Node.js
* npm
* PyTorch
* Torchvision
* Flask
* OpenCV
* NumPy
* Pillow
* Scikit-learn

The final model can perform inference on a CPU.

---

# Backend Setup

Create and activate a Python virtual environment.

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

Install the required Python packages:

```powershell
pip install torch torchvision flask pillow opencv-python numpy scikit-learn
```

Navigate to the backend directory and start the Flask server.

For example:

```powershell
python app.py
```

The backend runs locally and exposes the prediction endpoint:

```text
/predict
```

---

# Frontend Setup

Navigate to the frontend directory:

```powershell
cd frontend
```

Install the project dependencies:

```powershell
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

The terminal will provide the local address at which the frontend can be accessed.

---

# API

The backend provides a `/predict` endpoint for image classification.

### Request

```http
POST /predict
Content-Type: multipart/form-data
```

The image should be provided using the form field:

```text
image
```

### Example Response

```json
{
  "prediction": "synthetic",
  "confidence": 0.95,
  "probabilities": {
    "real": 0.03,
    "synthetic": 0.95,
    "swapped": 0.02
  }
}
```

The exact probability values vary depending on the input image.

---

# Model Training

The training pipeline can be used to train the EfficientNet-B0 model using the prepared dataset.

From the appropriate project directory:

```powershell
python src\train_model2.py
```

The training pipeline:

1. Loads the prepared dataset
2. Creates the EfficientNet-B0 model
3. Loads ImageNet-pretrained weights
4. Replaces the original classifier
5. Trains the three-class model
6. Evaluates validation performance after each epoch
7. Saves epoch checkpoints
8. Saves the model with the highest validation accuracy

---

# Model Evaluation

The final model can be evaluated using:

```powershell
python src\test_model2.py
```

The evaluation script reports:

* Number of test images
* Selected model epoch
* Validation accuracy of the selected model
* Test accuracy
* Confusion matrix
* Precision
* Recall
* F1-score

The test dataset is not used during model training or model selection.

---

# Computational Environment

The system was developed and trained using a laptop-based CPU environment.

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

# Academic Context

This project was developed as a **final-year project at the University of Ghana, Legon** within the field of Computer Science.

The project investigates the use of convolutional neural networks and transfer learning for automated deepfake image classification.

The final system demonstrates an end-to-end workflow covering:

```text
Dataset Preparation
       ↓
Face Preprocessing
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Backend Integration
       ↓
Web Application
       ↓
User Prediction
```

---

# Acknowledgements

The project makes use of publicly distributed datasets and open-source machine-learning and software-development technologies.

The dataset sources, software libraries, and frameworks used in the project are acknowledged for supporting the development and evaluation of the system.

---

# Author

**Vanessa Elinam Daker**

University of Ghana, Legon

---

# License

This project was developed for academic purposes as part of a final-year project.

Individual datasets and third-party libraries used by the project may be subject to their respective licenses and terms of use.
