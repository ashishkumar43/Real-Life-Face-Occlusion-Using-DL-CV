
#hat + mask only

import os
import cv2
import random
import numpy as np
from PIL import Image
import face_alignment

# Load face-alignment model (GPU)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

image_counter = 0

# Paths
input_folder = 'images'
hat_folder = 'Accessories/Hats'
mask_folder = 'Accessories/Masks'
output_hat_mask = 'output/hat_mask'

os.makedirs(output_hat_mask, exist_ok=True)

accessory_offsets = {
    # Hats
    'hat1.png': (5, -25),
    'hat2.png': (4.8, -33),
    'hat3.png': (7, -24),
    'hat4.png': (5, -27),
    'hat5.png': (9, -37),
    'hat6.png': (10, -18),
    'hat7.png': (7, -11),
    'hat8.png': (4, -18),
    'hat9.png': (6, -18),
    'hat10.png': (8, -18),
    'hat11.png': (6, -18),
    'hat12.png': (3, -18),
    'hat13.png': (5, -18),
    'hat14.png': (4, -18),
    
    'mask1.png': (0, -15),
    'mask2.png': (2, -10),
    'mask3.png': (2, -10),
    'mask4.png': (2, -10),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_hat_mask(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    landmarks = get_landmarks(img_np)
    if landmarks is None:
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
    hat_w = int(face_width * 1.3)
    hat_h = int(face_height * 1.52)
    hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
    hat_y = int(forehead[1] - hat_h + hat_offset_y)
    face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

    # Mask
    mask_name = random.choice(os.listdir(mask_folder))
    mask_path = os.path.join(mask_folder, mask_name)
    mask_img = Image.open(mask_path)
    mask_offset_x, mask_offset_y = accessory_offsets.get(mask_name, (0, -10))

    nose = landmarks[30]
    chin = landmarks[8]
    mask_h = int(np.linalg.norm(nose - chin) * 1.3)
    mask_w = int(mask_h * 1.5)
    mask_x = int(nose[0] - mask_w // 2 + mask_offset_x)
    mask_y = int(nose[1] + mask_offset_y)
    face_img = overlay_accessory(face_img, mask_img, mask_x, mask_y, mask_w, mask_h)

    # Replace filename logic to make outputs continuous
    global image_counter
    image_counter += 1
    filename = f"{image_counter:05}.jpg"  # 00001.jpg, 00002.jpg, ...
    face_img.convert('RGB').save(os.path.join(out_folder, filename))

# Process images
for file in os.listdir(input_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(input_folder, file)
        place_hat_mask(path)

print("✅ Done: hat + mask applied.")
