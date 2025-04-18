print("ashish kumar")

# import os
# import cv2
# import random
# import numpy as np
# from PIL import Image
# import face_alignment

# # Load face-alignment model (GPU)
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

# # Paths
# input_folder = 'images'
# hat_folder = 'Accessories/Hats'
# mask_folder = 'Accessories/Masks'
# glass_folder = 'Accessories/Sunglasses'

# output_hat_mask = 'output/hat_mask'
# output_hat_glass = 'output/hat_glass'

# os.makedirs(output_hat_mask, exist_ok=True)
# os.makedirs(output_hat_glass, exist_ok=True)

# def get_landmarks(image):
#     preds = fa.get_landmarks(image)
#     return preds[0] if preds else None

# def overlay_accessory(face_img, accessory_img, x, y, w, h):
#     accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS)
#     accessory = accessory.convert("RGBA")
#     face_img.paste(accessory, (x, y), accessory)
#     return face_img

# def place_accessories(image_path, mode='hat_mask'):
#     img = Image.open(image_path).convert('RGB')
#     img_np = np.array(img)
#     landmarks = get_landmarks(img_np)
#     if landmarks is None:
#         return

#     face_img = img.convert('RGBA')
#     landmarks = np.array(landmarks)

#     # Load Hat
#     hat_path = os.path.join(hat_folder, random.choice(os.listdir(hat_folder)))
#     hat_img = Image.open(hat_path)

#     chin = landmarks[8]
#     forehead = landmarks[27]
#     face_height = int(np.linalg.norm(chin - forehead))
#     face_width = int(np.linalg.norm(landmarks[0] - landmarks[16]))

#     hat_w = int(face_width * 1.4)
#     hat_h = int(face_height * 0.9)
#     # hat_x = int(forehead[0] - hat_w // 2)
#     # hat_y = int(forehead[1] - hat_h)
#     # face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

#     # Tune these manually
#     hat_offset_x = 8     # try -10, +10
#     hat_offset_y = -40     # try -20 to +20

#     hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
#     hat_y = int(forehead[1] - hat_h + hat_offset_y)

#     face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)
    
#     if mode == 'hat_glass':
#         # Load sunglasses
#         glass_path = os.path.join(glass_folder, random.choice(os.listdir(glass_folder)))
#         glass_img = Image.open(glass_path)

#         left_eye = np.mean(landmarks[36:42], axis=0)
#         right_eye = np.mean(landmarks[42:48], axis=0)
#         eye_center = ((left_eye + right_eye) / 2).astype(int)
#         eye_width = int(np.linalg.norm(right_eye - left_eye) * 2.3)
#         eye_height = int(eye_width * 0.48)

#         glass_x = int(eye_center[0] - eye_width // 2)
#         glass_y = int(eye_center[1] - eye_height // 2)
#         face_img = overlay_accessory(face_img, glass_img, glass_x, glass_y, eye_width, eye_height)
        
#         glass_offset_x = 0     # try -10, +10
#         glass_offset_y = -5    # try -10 to +10

#         glass_x = int(eye_center[0] - eye_width // 2 + glass_offset_x)
#         glass_y = int(eye_center[1] - eye_height // 2 + glass_offset_y)

#         out_folder = output_hat_glass

#     elif mode == 'hat_mask':
#         # Load mask
#         mask_path = os.path.join(mask_folder, random.choice(os.listdir(mask_folder)))
#         mask_img = Image.open(mask_path)

#         nose = landmarks[30]
#         chin = landmarks[8]
#         mask_h = int(np.linalg.norm(nose - chin) * 1.3)
#         mask_w = int(mask_h * 1.5)
#         mask_x = int(nose[0] - mask_w // 2)
#         mask_y = int(nose[1])
#         face_img = overlay_accessory(face_img, mask_img, mask_x, mask_y, mask_w, mask_h)

#         mask_x = int(nose[0] - mask_w // 2)
#         mask_y = int(nose[1])
        
#         mask_offset_x = 0       # try -10, +10
#         mask_offset_y = -10     # try -20 to +10

#         mask_x = int(nose[0] - mask_w // 2 + mask_offset_x)
#         mask_y = int(nose[1] + mask_offset_y)


#         out_folder = output_hat_mask

#     # Save result
#     filename = os.path.basename(image_path)
#     face_img.convert('RGB').save(os.path.join(out_folder, filename))

# # 🔁 Process images
# for file in os.listdir(input_folder):
#     if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#         path = os.path.join(input_folder, file)
#         mode = random.choice(['hat_mask', 'hat_glass'])  # Only two modes
#         place_accessories(path, mode)

# print("✅ Done: Accurate accessories applied.")



# import os
# import cv2
# import random
# import numpy as np
# from PIL import Image
# import face_alignment

# # Load face-alignment model (GPU)
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

# # Paths
# input_folder = 'images'
# hat_folder = 'Accessories/Hats'
# mask_folder = 'Accessories/Masks'
# glass_folder = 'Accessories/Sunglasses'

# output_hat_mask = 'output/hat_mask'
# output_hat_glass = 'output/hat_glass'

# os.makedirs(output_hat_mask, exist_ok=True)
# os.makedirs(output_hat_glass, exist_ok=True)

# # 🧩 Add this just below your output folders:
# accessory_offsets = {
#     # Hats
#     'hat1.png': (5, -30),
#     'hat2.png': (0, -10),
#     'hat3.png': (3, -20),
#     'hat4.png': (5, -30),
#     'hat5.png': (8, -30),
#     'hat6.png': (10, -20),
#     'hat7.png': (3, -15),
#     'hat8.png': (6, -17),
#     # Masks
#     'mask1.png': (0, -15),
#     'mask2.png': (2, -10),
#     'mask3.png': (2, -10),
#     'mask4.png': (2, -10),
#     # Glasses
#     'glass1.png': (0, -8),
#     'glass2.png': (-3, -5),
#     'glass3.png': (0, -8),
#     'glass4.png': (0, -8),
#     'glass5.png': (0, -8),
#     'glass6.png': (0, -8),
#     'glass7.png': (0, -8),
#     'glass8.png': (0, -8),
#     'glass9.png': (0, -8),
#     'glass10.png': (0, -8),
#     'glass11.png': (0, -8),
# }

# def get_landmarks(image):
#     preds = fa.get_landmarks(image)
#     return preds[0] if preds else None

# def overlay_accessory(face_img, accessory_img, x, y, w, h):
#     accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS)
#     accessory = accessory.convert("RGBA")
#     face_img.paste(accessory, (x, y), accessory)
#     return face_img

# def place_accessories(image_path, mode='hat_mask'):
#     img = Image.open(image_path).convert('RGB')
#     img_np = np.array(img)
#     landmarks = get_landmarks(img_np)
#     if landmarks is None:
#         return

#     face_img = img.convert('RGBA')
#     landmarks = np.array(landmarks)

#     # # Load Hat
#     # hat_path = os.path.join(hat_folder, random.choice(os.listdir(hat_folder)))
#     # hat_img = Image.open(hat_path)
    
#     hat_name = random.choice(os.listdir(hat_folder))
#     hat_path = os.path.join(hat_folder, hat_name)
#     hat_img = Image.open(hat_path)
    
#     # Offset lookup
#     hat_offset_x, hat_offset_y = accessory_offsets.get(hat_name, (7, -35))


#     chin = landmarks[8]
#     forehead = landmarks[27]
#     face_height = int(np.linalg.norm(chin - forehead))
#     face_width = int(np.linalg.norm(landmarks[0] - landmarks[16]))

#     # hat_w = int(face_width * 1.2)
#     # hat_h = int(face_height * 1.5)
#     # hat_x = int(forehead[0] - hat_w // 2)
#     # hat_y = int(forehead[1] - hat_h)
#     # face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

#     # # Tune these manually
#     # hat_offset_x = 7     # try -10, +10
#     # hat_offset_y = -35     # try -20 to +20

#     hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
#     hat_y = int(forehead[1] - hat_h + hat_offset_y)

#     face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)
    
#     if mode == 'hat_glass':
#         # # Load sunglasses
#         # glass_path = os.path.join(glass_folder, random.choice(os.listdir(glass_folder)))
#         # glass_img = Image.open(glass_path)
        
#         glass_name = random.choice(os.listdir(glass_folder))
#         glass_path = os.path.join(glass_folder, glass_name)
#         glass_img = Image.open(glass_path)

#         # Offset lookup
#         glass_offset_x, glass_offset_y = accessory_offsets.get(glass_name, (0, -5))

        

#         left_eye = np.mean(landmarks[36:42], axis=0)
#         right_eye = np.mean(landmarks[42:48], axis=0)
#         eye_center = ((left_eye + right_eye) / 2).astype(int)
#         eye_width = int(np.linalg.norm(right_eye - left_eye) * 2.3)
#         eye_height = int(eye_width * 0.48)

#         # glass_x = int(eye_center[0] - eye_width // 2)
#         # glass_y = int(eye_center[1] - eye_height // 2)
#         # face_img = overlay_accessory(face_img, glass_img, glass_x, glass_y, eye_width, eye_height)
        
#         # glass_offset_x = 0     # try -10, +10
#         # glass_offset_y = -5    # try -10 to +10

#         # glass_x = int(eye_center[0] - eye_width // 2 + glass_offset_x)
#         # glass_y = int(eye_center[1] - eye_height // 2 + glass_offset_y)

#         glass_x = int(eye_center[0] - eye_width // 2 + glass_offset_x)
#         glass_y = int(eye_center[1] - eye_height // 2 + glass_offset_y)
        
#         out_folder = output_hat_glass

#     elif mode == 'hat_mask':
#         # Load mask
#         mask_path = os.path.join(mask_folder, random.choice(os.listdir(mask_folder)))
#         mask_img = Image.open(mask_path)

#         nose = landmarks[30]
#         chin = landmarks[8]
#         mask_h = int(np.linalg.norm(nose - chin) * 1.3)
#         mask_w = int(mask_h * 1.5)
#         mask_x = int(nose[0] - mask_w // 2)
#         mask_y = int(nose[1])
#         face_img = overlay_accessory(face_img, mask_img, mask_x, mask_y, mask_w, mask_h)

#         mask_x = int(nose[0] - mask_w // 2)
#         mask_y = int(nose[1])
        
#         mask_offset_x = 0       # try -10, +10
#         mask_offset_y = -10     # try -20 to +10

#         mask_x = int(nose[0] - mask_w // 2 + mask_offset_x)
#         mask_y = int(nose[1] + mask_offset_y)


#         out_folder = output_hat_mask

#     # Save result
#     filename = os.path.basename(image_path)
#     face_img.convert('RGB').save(os.path.join(out_folder, filename))

# # 🔁 Process images
# for file in os.listdir(input_folder):
#     if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#         path = os.path.join(input_folder, file)
#         mode = random.choice(['hat_mask', 'hat_glass'])  # Only two modes
#         place_accessories(path, mode)

# print("✅ Done: Accurate accessories applied.")




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
glass_folder = 'Accessories/Sunglasses'

output_hat_mask = 'output/hat_mask'
output_hat_glass = 'output/hat_glass'

os.makedirs(output_hat_mask, exist_ok=True)
os.makedirs(output_hat_glass, exist_ok=True)

# 🧩 Offsets per accessory file
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
    # Masks
    'mask1.png': (0, -15),
    'mask2.png': (2, -10),
    'mask3.png': (2, -10),
    'mask4.png': (2, -10),
    # # Glasses
    # 'glass1.png': (0, -8),
    # 'glass2.png': (-3, -5),
    # 'glass3.png': (0, -8),
    # 'glass4.png': (0, -8),
    # 'glass5.png': (0, -8),
    # 'glass6.png': (0, -8),
    # 'glass7.png': (0, -8),
    # 'glass8.png': (0, -8),
    # 'glass9.png': (0, -8),
    # 'glass10.png': (0, -8),
    # 'glass11.png': (0, -8),
}

def get_landmarks(image):
    preds = fa.get_landmarks(image)
    return preds[0] if preds else None

def overlay_accessory(face_img, accessory_img, x, y, w, h):
    accessory = accessory_img.resize((w, h), Image.Resampling.LANCZOS)
    accessory = accessory.convert("RGBA")
    face_img.paste(accessory, (x, y), accessory)
    return face_img

def place_accessories(image_path, mode='hat_mask'):
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

    # 🎩 Hat
    hat_name = random.choice(os.listdir(hat_folder))
    hat_path = os.path.join(hat_folder, hat_name)
    hat_img = Image.open(hat_path)

    hat_offset_x, hat_offset_y = accessory_offsets.get(hat_name, (7, -35))

    hat_w = int(face_width * 1.3)
    hat_h = int(face_height * 1.52)
    hat_x = int(forehead[0] - hat_w // 2 + hat_offset_x)
    hat_y = int(forehead[1] - hat_h + hat_offset_y)

    face_img = overlay_accessory(face_img, hat_img, hat_x, hat_y, hat_w, hat_h)

    if mode == 'hat_glass':
        # 😎 Glasses
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
        out_folder = output_hat_glass

    elif mode == 'hat_mask':
        # 😷 Mask
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
        out_folder = output_hat_mask

    # 💾 Save result
    global image_counter
    image_counter += 1
    filename = f"{image_counter:05}.jpg"  
    face_img.convert('RGB').save(os.path.join(out_folder, filename))

# 🔁 Process all images
for file in os.listdir(input_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(input_folder, file)
        mode = random.choice(['hat_mask', 'hat_glass'])  # Only two modes
        place_accessories(path, mode)

print("✅ Done: Accurate accessories applied.")

