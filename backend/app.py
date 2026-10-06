from flask import Flask, request
from flask_cors import CORS
import os
import torch 
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn

NUM_CLASSES = 3
DEVICE = "cpu"

model = efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    1280,
    NUM_CLASSES
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models (experiment two)",
    "efficientnet_b0.pth"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict( # loading the weights from the best saved model
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval() # no learning, just predictions

inference_transform = transforms.Compose( # transformations from validation and evaluation, with added resizing of images.
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

# ==============
# Face Detection
# ==============

face_detector = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml" )

app = Flask(__name__) # creates flask application

CORS(app, origins=["https://erudite-app.vercel.app"]) # Allows deployed frontend to call API

@app.route("/") # Initial route named '/' sends GET request by default because we didn't specify
def home():
    return "Deepfake Detection Backend is running!" # message received when someone visits

@app.route("/predict", methods=['POST']) # route which is /predict, POST is a request
def predict():

    import time
    start_time = time.time()

    print("\nPrediction request received", flush=True)

    if "image" not in request.files:
        return {
            "error": "No image was provided"
        }, 400

    image = request.files["image"] # request helps us ask for something in order to do something

    if image.filename == "": # in case user didn't select a file but clicked button to show prediction.
        return {
        "error": "No image was selected"
    }, 400

    print(f"Received image: {image.filename}", flush=True)

    try:
        image = Image.open(image).convert("RGB")

        print(
            f"Image opened successfully: {time.time() - start_time:.2f}s",
            flush=True
        )

    except UnidentifiedImageError: # send an error instead of crashing if image is unsuitable in any way
        return {
            "error": "The uploaded file is not a valid image"
        }, 400

# ==============
# Face Detection 
# ==============
    
    try: 

        image_array = np.array(image) 

        print(
            f"Image converted to array: {time.time() - start_time:.2f}s",
            flush=True
        )
        
        gray_image = cv2.cvtColor(
            image_array, 
            cv2.COLOR_RGB2GRAY 
        ) 

        print(
            f"Image converted to grayscale: {time.time() - start_time:.2f}s",
            flush=True
        )

        print("Starting face detection...", flush=True)
        
        faces = face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30) 
        ) 

        print(
            f"Face detection completed: {time.time() - start_time:.2f}s",
            flush=True
        )
        
        if len(faces) == 0: 
            return { 
                "error": "No face was detected in the uploaded image. Please upload an image containing a visible face." 
            }, 400 
        
    except Exception as error: 
        print(f"Face detection error: {error}", flush=True) 
        
        return { 
            "error": "An error occurred while checking the image for a face"
        }, 500

    print("Starting image preprocessing...", flush=True)

    image_tensor = inference_transform(image)

    image_tensor = image_tensor.unsqueeze(0) # unsqueeze to add a batch dimension explaining that we're predicting only one image, not 8 like during training. so from [3, 224, 224] to [1, 3, 224, 224]

    image_tensor = image_tensor.to(DEVICE)

    print(
        f"Image preprocessing completed: {time.time() - start_time:.2f}s",
        flush=True
    )

    try:

        print("Starting model inference...", flush=True)

        with torch.no_grad(): # don't calculate gradients

            outputs = model(image_tensor) # images enters model and prediction is made, three scores, one for each class

        print(
            f"Model inference completed: {time.time() - start_time:.2f}s",
            flush=True
        )

        probabilities = torch.softmax(outputs, dim=1) # converts those scores into values that behave like probabilities

        predicted_class = torch.argmax( # which class has the highest probability? 0, 1, 2 for each class respectively
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][predicted_class].item() # gets the probability corresponding to the class the model selected

        class_names = [
            "real",
            "synthetic",
            "swapped"
        ]

        prediction = class_names[predicted_class]

        class_probabilities = {
            class_names[i]: probabilities[0][i].item()
            for i in range(NUM_CLASSES)
        }

    except Exception as error:
        print(f"Prediction error: {error}", flush=True)

        return {
            "error": "An error occurred while processing the image"
        }, 500

    print(
        f"Prediction completed successfully: {time.time() - start_time:.2f}s",
        flush=True
    )

    return {
    "prediction": prediction,
    "confidence": confidence,
    "probabilities": class_probabilities
    }

if __name__ == "__main__": # if we're running this file directly, start Flask server
    app.run()