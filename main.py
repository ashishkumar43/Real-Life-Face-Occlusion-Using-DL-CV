# main.py
import os
import requests
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIG ---
# Public Hugging Face model link
MODEL_URL = "https://huggingface.co/Ashish43/face-occlusion-model/resolve/main/efficientnetB1_model.h5"
MODEL_PATH = "efficientnetB1_model.h5"

# --- Download model if not exists ---
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Hugging Face (public)...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded successfully.")

# --- Load model ---
print("⏳ Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

# --- Class names (in same order as training) ---
class_names = ["Masked", "Unmasked", "Partially_Occluded"]

# --- Flask app setup ---
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save uploaded image
            image = Image.open(file).convert("RGB")
            os.makedirs("static", exist_ok=True)
            image_path = os.path.join("static", "uploaded_image.png")
            image.save(image_path)
            uploaded_image_url = "/" + image_path

            # Preprocess image
            resized_img = image.resize((240, 240))
            img_array = keras_image.img_to_array(resized_img)
            img_array = np.expand_dims(preprocess_input(img_array), axis=0)

            # Predict
            preds = model.predict(img_array)
            idx = int(np.argmax(preds))
            prediction = class_names[idx]
            confidence = round(float(np.max(preds) * 100), 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_image=uploaded_image_url
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
