# Real-Life Face Occlusion Using Deep Learning & Computer Vision

## Overview

**Real-Life Face Occlusion Using DL & CV** is a major project focused on detecting and handling **occlusions in human faces**, such as masks, hats, or glasses. The system leverages **Deep Learning** and **Computer Vision** techniques to improve face recognition performance in real-world scenarios where occlusions are common.

---

## Features

* Detect faces from images or videos in real-time.
* Identify and classify occlusions: masks, hats, glasses, etc.
* Handle multiple faces simultaneously.
* Improve downstream face recognition and analysis tasks under occlusion conditions.

---

## Repository Structure

```
Real-Life-Face-Occlusion-Using-DL-CV/
│
├── data/                  # Dataset with faces and occlusion images
├── models/                # Trained deep learning models
├── notebooks/             # Jupyter notebooks for training and experimentation
├── scripts/               # Python scripts for training and inference
├── app.py                 # Real-time Face Occlusion Detection App
├── face-occlusiontrain.ipynb  # Final model training notebook
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Dataset

* Based on **CelebHQ** and custom datasets.
* Includes synthetic occlusions like masks, glasses, and hats.
* Images resized to **218x178** pixels for model training.
* Annotated with occlusion types and positions.

---

## Technology Stack

* **Programming Language:** Python 3.x
* **Deep Learning Framework:** PyTorch / TensorFlow
* **Computer Vision Libraries:** OpenCV, Pillow
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ashishkumar43/Real-Life-Face-Occlusion-Using-DL-CV.git
cd Real-Life-Face-Occlusion-Using-DL-CV
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python scripts/train.py --data_path ./data --model_path ./models
```

### Inference / Testing

```bash
python app.py --image_path ./data/test_image.jpg --model_path ./models/face_occlusion_model.pth
```

* Detects faces and highlights occluded regions.

---

## Applications

* Face recognition systems in public spaces (mask compliance).
* Security and surveillance.
* AR/VR filters for virtual masks or glasses.
* Healthcare monitoring.

---

## Future Enhancements

* Real-time video stream support.
* Additional occlusion types (scarves, hands, objects).
* Web or mobile deployment.
* Integration with face recognition pipelines.

---

## Authors

**Harsh Kumar Jha** and **Ashish Kumar**

---
