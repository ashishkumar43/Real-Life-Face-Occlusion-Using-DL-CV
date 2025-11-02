# app.py
from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Force CPU only (Render doesn't use GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# ✅ Correct model path (you saved as .h5)
MODEL_PATH = "efficientnetB1_model.h5"

# Load model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file '{MODEL_PATH}' not found! Make sure it's in the same folder as app.py."
    )

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ✅ Define class names (must match your training order)
TRAIN_PATH = "train"
if os.path.exists(TRAIN_PATH):
    class_names = sorted(os.listdir(TRAIN_PATH))
else:
    # Manually define class names if train folder not available in Render
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
            os.makedirs("static", exist_ok=True)
            image.save("static/uploaded_image.png")
            uploaded_image_url = "/static/uploaded_image.png"

            # ✅ Preprocess
            resized_img = image.resize((240, 240))
            img_array = keras_image.img_to_array(resized_img)
            img_array_exp = np.expand_dims(preprocess_input(img_array), axis=0)

            # ✅ Predict
            pred = model.predict(img_array_exp)
            predicted_class_idx = np.argmax(pred)
            prediction = class_names[predicted_class_idx]
            confidence = float(np.max(pred) * 100)

            # Optional: Display sample class image if available
            predicted_class_dir = os.path.join(TRAIN_PATH, prediction)
            if os.path.exists(predicted_class_dir):
                files_in_class = [
                    f for f in os.listdir(predicted_class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if files_in_class:
                    img = Image.open(os.path.join(predicted_class_dir, files_in_class[0]))
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
