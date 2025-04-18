print("ashish kumar")
# Combination hat + sunglasses only
import os
import cv2
import random
import numpy as np
from PIL import Image
import face_alignment

# Load face-alignment model (GPU)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

input_folder = 'images'
hat_folder = 'Accessories/Hats'
glass_folder = 'Accessories/Sunglasses'
output_hat_glass = 'output/hat_glass'

os.makedirs(output_hat_glass, exist_ok=True)

# Offsets
accessory_offsets = {
    'hat1.png': (5.4, -23),
    'hat2.png': (4.8, -36),
    'hat3.png': (6, -27),
    'hat4.png': (5, -26),
    'hat5.png': (9, -37),
    'hat6.png': (10, -18),
    'hat7.png': (6, -11),
    'hat8.png': (5, -14),
    'hat9.png': (6, -22),
    'hat10.png': (8.5, -16),
    'hat11.png': (5.5, -18),
    'hat12.png': (3, -17),
    'hat13.png': (5, -18),
    'hat14.png': (4, -15),
    'hat15.png': (3.5, -20),
    'hat16.png': (4, -22),
    'hat17.png': (2, -38),
    'hat18.png': (4.5, -25),
    'hat19.png': (4.5, -25),
    'hat20.png': (4, -25),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_hat_glass(image_path, index):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    landmarks = get_landmarks(img_np)
    if landmarks is None:
        print(f"No landmarks found: {image_path}")
        return

    face_img = img.convert('RGBA')
    landmarks = np.array(landmarks)

    chin = landmarks[8]
    forehead = landmarks[27]
    face_height = int(np.linalg.norm(chin - forehead))
    face_width = int(np.linalg.norm(landmarks[0] - landmarks[16]))

    # Hat
    hat_name = random.choice(os.listdir(hat_folder))
    hat_path = os.path.join(hat_folder, hat_name)
    hat_img = Image.open(hat_path)
    hat_offset_x, hat_offset_y = accessory_offsets.get(hat_name, (7, -35))
    hat_w = int(face_width * 1.5)
    hat_h = int(face_height * 1.52)
    hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
    hat_y = int(forehead[1] - hat_h + hat_offset_y)
    face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

    # Glasses
    glass_name = random.choice(os.listdir(glass_folder))
    glass_path = os.path.join(glass_folder, glass_name)
    glass_img = Image.open(glass_path)
    glass_offset_x, glass_offset_y = accessory_offsets.get(glass_name, (0, -5))

    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    eye_center = ((left_eye + right_eye) / 2).astype(int)
    eye_width = int(np.linalg.norm(right_eye - left_eye) * 2.3)
    eye_height = int(eye_width * 0.48)

    glass_x = int(eye_center[0] - eye_width // 2 + glass_offset_x)
    glass_y = int(eye_center[1] - eye_height // 2 + glass_offset_y)
    face_img = overlay_accessory(face_img, glass_img, glass_x, glass_y, eye_width, eye_height)

    # Save output
    filename = f"{index:05}.jpg"
    face_img.convert('RGB').save(os.path.join(output_hat_glass, filename))
    print(f"save_img: {filename}")


# Process all images in increasing order
image_counter = 0
for file in sorted(os.listdir(input_folder)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(input_folder, file)
        image_counter += 1
        place_hat_glass(path, image_counter)

print(" All hat + sunglasses images done.")