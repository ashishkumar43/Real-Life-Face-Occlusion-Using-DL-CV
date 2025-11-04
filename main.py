import streamlit as st
import numpy as np
# app.py
from flask import Flask, render_template, request
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="😷 Face Occlusion Classification",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="collapsed"
)
app = Flask(__name__)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        color: white;
        background: #6336e0;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #5a2bd1;
        transform: scale(1.03);
    }
    .result-box {
        padding: 1em;
        margin-top: 20px;
        border-radius: 10px;
        background: #262730;
        color: #ffffff;
        font-size: 1.2em;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
MODEL_PATH = "efficientnetB1_model.h5"
GDRIVE_FILE_ID = "14EjHLSK1e19J-L6I4Xu4bMFnTqsliWlw"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_resource
# Load model
def load_occlusion_model():
    model_path = "efficientnetB1_model.h5"
    gdrive_file_id = "14EjHLSK1e19J-L6I4Xu4bMFnTqsliWlw"
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... ⏳"):
            gdown.download(gdrive_url, model_path, quiet=False)

    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_occlusion_model()

TRAIN_PATH = "train"  

TRAIN_PATH = "train"
if os.path.exists(TRAIN_PATH):
    class_names = sorted(os.listdir(TRAIN_PATH))
else:
    class_names = ["Masked", "Unmasked", "Partially_Occluded"]  

st.title("😷 Face Occlusion Classification")
st.markdown("### Please upload an occluded face image to Begin classification")
st.markdown("✨ Model: EfficientNetB1 trained on occluded face dataset")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    st.markdown("#### ✅ Processing...")

    resized_img = uploaded_image.resize((240, 240))
    img_array = keras_image.img_to_array(resized_img)
    img_array_exp = np.expand_dims(preprocess_input(img_array), axis=0)

    with st.spinner("Running prediction..."):
        pred = model.predict(img_array_exp)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100
        label_name = class_names[predicted_class]

    predicted_class_dir = os.path.join(TRAIN_PATH, label_name)
    predicted_image_path = None
    if os.path.exists(predicted_class_dir):
        files_in_class = [
            f for f in os.listdir(predicted_class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if files_in_class:
            predicted_image_path = os.path.join(predicted_class_dir, files_in_class[0])

    st.markdown(f"""
    <div class="result-box" style="border-left: 5px solid #36d7b7;">
        <b>Predicted Identity:</b> {label_name}<br>
        <b>Confidence:</b> {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Visual Output")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(uploaded_image, caption=f"Predicted: {label_name}\nConfidence: {confidence:.2f}%", use_container_width=True)
    with col3:
        if predicted_image_path:
            st.image(Image.open(predicted_image_path), caption=f"Actual Class: {label_name}", use_container_width=True)
        else:
            st.markdown("No class reference image found", unsafe_allow_html=True)
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