print("ashish kumar")
# Combination mask + sunglasses only
import os
import numpy as np
from PIL import Image
import face_alignment

# Load face-alignment model (GPU)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

input_folder = 'images'
mask_folder = 'Accessories/Masks'
glass_folder = 'Accessories/Sunglasses'
output_glass_mask = 'output/glass_mask'

os.makedirs(output_glass_mask, exist_ok=True)

# Offsets
accessory_offsets = {
    'mask2.png': (0, 0),
    'mask3.png': (0, 1),
    'mask4.png': (5, -12),
    'mask5.png': (-4.6, -5),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_glass_mask(image_path, glass_name, mask_name, index):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    landmarks = get_landmarks(img_np)
    if landmarks is None:
        print(f"No landmarks found: {image_path}")
        return

    face_img = img.convert('RGBA')
    landmarks = np.array(landmarks)

    face_height = int(np.linalg.norm(landmarks[8] - landmarks[27]))
    face_width = int(np.linalg.norm(landmarks[0] - landmarks[16]))

    # Glasses
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

    # Mask
    mask_path = os.path.join(mask_folder, mask_name)
    mask_img = Image.open(mask_path)
    mask_offset_x, mask_offset_y = accessory_offsets.get(mask_name, (0, 0))

    nose = landmarks[33]
    jaw_left = landmarks[3]
    jaw_right = landmarks[13]
    mask_center = nose
    mask_w = int(np.linalg.norm(jaw_right - jaw_left) * 1.8)
    mask_h = int(face_height * 1.45)

    mask_x = int(mask_center[0] - mask_w // 2 + mask_offset_x)
    mask_y = int(mask_center[1] - mask_h // 3 + mask_offset_y)
    face_img = overlay_accessory(face_img, mask_img, mask_x, mask_y, mask_w, mask_h)

    # Save output
    filename = f"{index:05}.jpg"
    face_img.convert('RGB').save(os.path.join(output_glass_mask, filename))
    print(f"save_img: {filename}")

# Lists
glass_list = [f'glass{i}.png' for i in range(1,11)]
mask_list = [f'mask{i}.png' for i in range(1,6)]

input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for i, file in enumerate(input_files):
    glass_name = glass_list[i % len(glass_list)]
    mask_name = mask_list[i % len(mask_list)]
    img_path = os.path.join(input_folder, file)
    place_glass_mask(img_path, glass_name, mask_name, i)

print("All mask + sunglasses images done.")
