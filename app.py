import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from flask import Flask, request, jsonify, render_template


# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS, "cnn_gwo_model.h5")
FEATURES_PATH = os.path.join(ARTIFACTS, "selected_features.npy")

# -----------------------
# Class labels (ORDER MUST MATCH TRAINING)
# -----------------------
CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# -----------------------
# Load selected feature indices
# -----------------------
# Load selected features
selected_idx = np.load(FEATURES_PATH)[:240]
NUM_FEATURES = 240
print(f"Loaded selected features: {NUM_FEATURES}")

# Rebuild classifier
classifier = Sequential([
    Input(shape=(240,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(4, activation="softmax")
])

# Load weights (THIS IS STEP 3)
classifier.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
print("Classifier reconstructed & weights loaded")

# -----------------------
# Load VGG16 feature extractor
# -----------------------
vgg = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

feature_extractor = Model(
    inputs=vgg.input,
    outputs=vgg.output
)
print("VGG16 feature extractor loaded")

# -----------------------
# MRI validation (DO NOT CHANGE)
# -----------------------
def is_mri_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    std = np.std(gray)
    mean = np.mean(gray)

    if std < 15:
        return False
    if mean < 40 or mean > 220:
        return False

    return True

# -----------------------
# Feature extraction
# -----------------------
def extract_features(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feats = feature_extractor.predict(img, verbose=0)
    feats = feats.flatten()

    feats = feats[selected_idx]
    return feats.reshape(1, -1)

# -----------------------
# Routes
# -----------------------
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # MRI check
    if not is_mri_image(img):
        return jsonify({
            "prediction": "Invalid image",
            "message": "Please upload a valid brain MRI scan"
        }), 200

    # Extract features
    features = extract_features(img)

    # Predict
    probs = classifier.predict(features, verbose=0)[0]
    idx = int(np.argmax(probs))

    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidence": float(probs[idx])
    })

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
