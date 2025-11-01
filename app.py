# app.py
from flask import Flask, render_template, request
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

MODEL_PATH = "efficientnetB1_model.h5"
GDRIVE_FILE_ID = "14EjHLSK1e19J-L6I4Xu4bMFnTqsliWlw"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Load model
def load_occlusion_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_occlusion_model()

TRAIN_PATH = "train"
if os.path.exists(TRAIN_PATH):
    class_names = sorted(os.listdir(TRAIN_PATH))
else:
    class_names = ["Masked", "Unmasked", "Partially_Occluded"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image_url = None
    predicted_image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file).convert("RGB")
            image.save("static/uploaded_image.png")  # save for display
            uploaded_image_url = "/static/uploaded_image.png"

            # Preprocess image
            resized_img = image.resize((240, 240))
            img_array = keras_image.img_to_array(resized_img)
            img_array_exp = np.expand_dims(preprocess_input(img_array), axis=0)

            # Prediction
            pred = model.predict(img_array_exp)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred) * 100
            prediction = class_names[predicted_class]

            predicted_class_dir = os.path.join(TRAIN_PATH, prediction)
            if os.path.exists(predicted_class_dir):
                files_in_class = [
                    f for f in os.listdir(predicted_class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if files_in_class:
                    img_path = os.path.join(predicted_class_dir, files_in_class[0])
                    img = Image.open(img_path)
                    img.save("static/predicted_image.png")
                    predicted_image_url = "/static/predicted_image.png"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           uploaded_image=uploaded_image_url,
                           predicted_image=predicted_image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
