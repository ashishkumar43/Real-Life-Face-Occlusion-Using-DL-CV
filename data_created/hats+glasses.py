print("ashish kumar")
# Combination hat + sunglasses only
import os
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
    'hat1.png': (5.4, -10),
    'hat2.png': (4.8, -52),
    'hat3.png': (5.2, -25),
    'hat4.png': (6.6, -26),
    'hat5.png': (17, -50),
    'hat6.png': (9, -11),
    'hat7.png': (6, -12),
    'hat8.png': (3, -50),
    'hat9.png': (1, -45),
    'hat10.png': (8.5, -20),
    'hat11.png': (5.3, -22),
    'hat12.png': (5.6, -45),
    'hat13.png': (4.5, -28),
    'hat14.png': (5.5, -12),
    'hat15.png': (4.2, -40),
    'hat16.png': (5.4, -30),
    'hat17.png': (1.6, -30),
    'hat18.png': (5, -48),
    'hat19.png': (2.5, -42),
    'hat20.png': (5.7, -32),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_hat_glass(image_path, hat_name, glass_name, index):
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
    hat_w = int(face_width * 1.6)
    hat_h = int(face_height * 1.52)
    hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
    hat_y = int(forehead[1] - hat_h + hat_offset_y)
    face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

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

    # Save output
    filename = f"{index:05}.jpg"
    face_img.convert('RGB').save(os.path.join(output_hat_glass, filename))
    print(f"save_img: {filename}")


# Get images in serial order
hat_list = [f'hat{i}.png' for i in range(1, 21)]
glass_list = [f'glass{i}.png' for i in range(1,11 )]

# Sorted input image list
input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# fixed pairs
for i, file in enumerate(input_files):
    hat_name = hat_list[i % len(hat_list)]
    glass_name = glass_list[i % len(glass_list)]
    img_path = os.path.join(input_folder, file)
    place_hat_glass(img_path, hat_name, glass_name, i)

print("All hat + sunglasses images done.")
