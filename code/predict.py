import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from LISA import LISA

COLOR_MAP = [
    [0, 0, 0],        # 0: background
    [204, 0, 0],      # 1: skin
    [76, 153, 0],     # 2: nose
    [204, 204, 0],    # 3: eye_g
    [51, 51, 255],    # 4: l_eye
    [204, 0, 204],    # 5: r_eye
    [0, 255, 255],    # 6: l_brow
    [255, 204, 204],  # 7: r_brow
    [102, 51, 0],     # 8: l_ear
    [255, 0, 0],      # 9: r_ear
    [102, 204, 0],    # 10: mouth
    [255, 255, 0],    # 11: u_lip
    [0, 0, 153],      # 12: l_lip
    [0, 0, 204],      # 13: hair
    [255, 51, 153],   # 14: hat
    [0, 204, 204],    # 15: ear_r
    [0, 51, 0],       # 16: neck_l
    [255, 153, 51],   # 17: neck
    [0, 204, 0],     
]

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(mask):
    return mask.squeeze().cpu().numpy()

def apply_color_map(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for class_idx, _ in enumerate(COLOR_MAP):
        color_mask[mask == class_idx] = class_idx
    return color_mask

def save_mask(mask, save_path):
    colored_mask = apply_color_map(mask)
    mask_image = Image.fromarray(colored_mask, mode='P')
    palette = [color for sublist in COLOR_MAP for color in sublist]
    mask_image.putpalette(palette)
    mask_image.save(save_path, format='PNG', optimize=True)

def save_aug(mask, save_path):
    mask_image = Image.fromarray(mask, mode='P')
    palette = [color for sublist in COLOR_MAP for color in sublist]
    mask_image.putpalette(palette)
    mask_image.save(save_path, format='PNG', optimize=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LISA().to(device)
    model.load_state_dict(torch.load('./best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    input_folder = '/data3/zhangsn/7454CW1/test_image'
    output_folder = 'output'

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processing images"):
            
            image_path = os.path.join(input_folder, image_file)
            image = load_image(image_path)
            input_tensor = preprocess_image(image).to(device)

            output = model(input_tensor)
            _, mask = torch.max(output, 1)

            mask = postprocess_mask(mask)

            save_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.png")
            save_mask(mask, save_path)

    print(f"Processed {len(image_files)} images. 8-bit colored masks saved in {output_folder}")

if __name__ == "__main__":
    main()