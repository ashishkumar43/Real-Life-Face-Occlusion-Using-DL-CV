# app.py
from flask import Flask, render_template, request
import os
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import load_model

# 1️⃣ Load your existing H5 model
h5_model_path = "efficientnetB1_model.h5"
model = load_model(h5_model_path, compile=False)  # use compile=False to avoid custom loss issues

# 2️⃣ Save it in the new .keras format
keras_model_path = "efficientnetB1_model.keras"
model.save(keras_model_path)

print(f"✅ Model converted and saved as {keras_model_path}")


app = Flask(__name__)

MODEL_PATH = "efficientnetB1_model.keras"  
GDRIVE_FILE_ID = "14EjHLSK1e19J-L6I4Xu4bMFnTqsliWlw"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Load model (downloads from GDrive if not exists)
def load_occlusion_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH, compile=False)
    return model

# Load model once at startup
model = load_occlusion_model()

# Define class names (must match your training order)
TRAIN_PATH = "train"
if os.path.exists(TRAIN_PATH):
    class_names = sorted(os.listdir(TRAIN_PATH))
else:
    # fallback if train directory doesn't exist
    class_names = ["Masked", "Unmasked", "Partially_Occluded"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image_url = None
    predicted_image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(file).convert("RGB")
            image.save("static/uploaded_image.png")
            uploaded_image_url = "/static/uploaded_image.png"

            # Preprocess
            resized_img = image.resize((240, 240))
            img_array = keras_image.img_to_array(resized_img)
            img_array_exp = np.expand_dims(preprocess_input(img_array), axis=0)

            # Prediction
            pred = model.predict(img_array_exp)
            predicted_class_idx = np.argmax(pred)
            prediction = class_names[predicted_class_idx]
            confidence = float(np.max(pred) * 100)

            # Optional: show a sample image from predicted class
            predicted_class_dir = os.path.join(TRAIN_PATH, prediction)
            if os.path.exists(predicted_class_dir):
                files_in_class = [
                    f for f in os.listdir(predicted_class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if files_in_class:
                    sample_img_path = os.path.join(predicted_class_dir, files_in_class[0])
                    img = Image.open(sample_img_path)
                    img.save("static/predicted_image.png")
                    predicted_image_url = "/static/predicted_image.png"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_image=uploaded_image_url,
        predicted_image=predicted_image_url
    )


if __name__ == "__main__":
    # force CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
