import os

import cv2
import numpy as np
import streamlit_app as st
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="◉",
    layout="centered"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

NUM_CLASSES = 3
DEVICE = "cpu"

CLASS_NAMES = ["real", "synthetic", "swapped"]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models (experiment two)",
    "efficientnet_b0.pth"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        1280,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# FACE DETECTOR
# ---------------------------------------------------------

@st.cache_resource
def load_face_detector():

    return cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------

try:

    model = load_model()
    face_detector = load_face_detector()

except Exception as error:

    st.error(
        "The detection model could not be loaded."
    )

    st.exception(error)

    st.stop()


# ---------------------------------------------------------
# PAGE CONTENT
# ---------------------------------------------------------

st.title("Deepfake Image Detection")

st.write(
    "Upload an image containing a visible face "
    "to determine whether it is likely to be "
    "real, face-swapped, or synthetic."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except UnidentifiedImageError:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()


    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )


    # Convert image to OpenCV format
    image_array = np.array(image)

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )


    # -----------------------------------------------------
    # FACE DETECTION
    # -----------------------------------------------------

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    if len(faces) == 0:

        st.error(
            "No face was detected in the uploaded image. "
            "Please upload an image containing a visible face."
        )

        st.stop()


    # -----------------------------------------------------
    # MODEL PREPROCESSING
    # -----------------------------------------------------

    image_tensor = inference_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)


    # -----------------------------------------------------
    # MODEL INFERENCE
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = probabilities[
        0
    ][predicted_class].item()


    prediction = CLASS_NAMES[
        predicted_class
    ]


    class_probabilities = {
        CLASS_NAMES[i]:
        probabilities[0][i].item()
        for i in range(NUM_CLASSES)
    }


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    display_labels = {
        "real": "LIKELY REAL",
        "synthetic": "LIKELY SYNTHETIC",
        "swapped": "LIKELY FACE-SWAPPED"
    }


    st.divider()

    st.subheader(
        display_labels[prediction]
    )


    st.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )


    st.write("Classification probabilities")


    st.progress(
        class_probabilities["real"],
        text=f"REAL — {class_probabilities['real'] * 100:.1f}%"
    )

    st.progress(
        class_probabilities["swapped"],
        text=f"FACE-SWAPPED — {class_probabilities['swapped'] * 100:.1f}%"
    )

    st.progress(
        class_probabilities["synthetic"],
        text=f"SYNTHETIC — {class_probabilities['synthetic'] * 100:.1f}%"
    )