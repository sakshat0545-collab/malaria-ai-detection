from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time

app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/malaria_efficientnet.h5"
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)

    return image


# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):

    processed = preprocess_image(image)

    start = time.time()
    prediction = model.predict(processed)[0][0]
    end = time.time()

    processing_time = round(end - start, 3)

    if prediction > 0.5:
        label = "Parasitized"
        confidence = prediction
    else:
        label = "Uninfected"
        confidence = 1 - prediction

    confidence = round(float(confidence) * 100, 2)

    return label, confidence, processing_time


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    processing_time = None
    filename = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")

            result, confidence, processing_time = predict(image)

            filename = "uploads/" + file.filename

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        processing_time=processing_time,
        filename=filename
    )


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)