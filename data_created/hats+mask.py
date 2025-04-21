print("ashish kumar")
# Combination hat + mask only
import os
import numpy as np
from PIL import Image
import face_alignment

# Load face-alignment model (GPU)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

input_folder = 'images'
hat_folder = 'Accessories/Hats'
mask_folder = 'Accessories/Masks'
output_hat_mask = 'output/hat_mask'

os.makedirs(output_hat_mask, exist_ok=True)

# Offsets
accessory_offsets = {
    'hat1.png': (5.4, -10),
    'hat2.png': (4.8, -55),
    'hat3.png': (5.6, -30),
    'hat4.png': (5.8, -26),
    'hat5.png': (10, -50),
    'hat6.png': (10, -11),
    'hat7.png': (6, -12),
    'hat8.png': (4, -16),
    'hat9.png': (3, -35),
    'hat10.png': (8.5, -16),
    'hat11.png': (5.3, -17.7),
    'hat12.png': (3, -30),
    'hat13.png': (5, -25),
    'hat14.png': (4.2, -20),
    'hat15.png': (4.5, -40),
    'hat16.png': (4.2, -25),
    'hat17.png': (1, -40),
    'hat18.png': (5, -36),
    'hat19.png': (3, -37),
    'hat20.png': (5, -30),

    # mask offsets 
    'mask1.png': (0, 0),
    'mask2.png': (0, 0),
    'mask3.png': (0, 0),
    'mask4.png': (0, 0),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_hat_mask(image_path, hat_name, mask_name, index):
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
    hat_path = os.path.join(hat_folder, hat_name)
    hat_img = Image.open(hat_path)
    hat_offset_x, hat_offset_y = accessory_offsets.get(hat_name, (7, -35))
    hat_w = int(face_width * 1.5)
    hat_h = int(face_height * 1.52)
    hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
    hat_y = int(forehead[1] - hat_h + hat_offset_y)
    face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

    # Mask
    mask_path = os.path.join(mask_folder, mask_name)
    mask_img = Image.open(mask_path)
    mask_offset_x, mask_offset_y = accessory_offsets.get(mask_name, (0, 0))

    nose = landmarks[33]
    jaw_left = landmarks[3]
    jaw_right = landmarks[13]
    mask_center = nose
    mask_w = int(np.linalg.norm(jaw_right - jaw_left) * 1.3)
    mask_h = int(face_height * 1)

    mask_x = int(mask_center[0] - mask_w // 2 + mask_offset_x)
    mask_y = int(mask_center[1] - mask_h // 3 + mask_offset_y)
    face_img = overlay_accessory(face_img, mask_img, mask_x, mask_y, mask_w, mask_h)

    # Save output
    filename = f"{index:05}.jpg"
    face_img.convert('RGB').save(os.path.join(output_hat_mask, filename))
    print(f"save_img: {filename}")

# Image and accessory lists
hat_list = [f'hat{i}.png' for i in range(1, 21)]
mask_list = [f'mask{i}.png' for i in range(1, 5)]

# Sorted input image list
input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Apply combination
for i, file in enumerate(input_files):
    hat_name = hat_list[i % len(hat_list)]
    mask_name = mask_list[i % len(mask_list)]
    img_path = os.path.join(input_folder, file)
    place_hat_mask(img_path, hat_name, mask_name, i)

print("All hat + mask images done.")


