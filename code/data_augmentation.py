import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from predict import postprocess_mask, save_aug
import os

class CustomTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.left_eye_color = 4
        self.right_eye_color = 5
        self.left_brow_color = 6
        self.right_brow_color = 7
        self.left_ear_color = 8
        self.right_ear_color = 9

    def __call__(self, image, mask):
        if self.is_train:
            # Horizontal flip
            image = TF.hflip(image)
            mask = self.flip_mask(mask)

            # Random rotation
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, fill=(0))
            mask = np.array(mask)

            # Color jitter
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        return image, mask

    def flip_mask(self, mask):
        # Convert mask to numpy array
        mask_np = np.array(mask)

        # Flip the mask
        flipped_mask = np.fliplr(mask_np)

        # Swap the left and right eye colors
        left_eye_mask = flipped_mask == self.left_eye_color
        right_eye_mask = flipped_mask == self.right_eye_color
        left_brow_mask = flipped_mask == self.left_brow_color
        right_brow_mask = flipped_mask == self.right_brow_color
        left_ear_mask = flipped_mask == self.left_ear_color
        right_ear_mask = flipped_mask == self.right_ear_color

        flipped_mask[left_eye_mask] = self.right_eye_color
        flipped_mask[right_eye_mask] = self.left_eye_color
        flipped_mask[left_brow_mask] = self.right_brow_color
        flipped_mask[right_brow_mask] = self.left_brow_color
        flipped_mask[left_ear_mask] = self.right_ear_color
        flipped_mask[right_ear_mask] = self.left_ear_color
        
        return flipped_mask.astype(np.uint8)

def process_augmentation(image_path, mask_path, num_augmentations=5):
    # Load original image and mask
    original_image = Image.open(image_path).convert('RGB')
    original_mask = Image.open(mask_path)

    # Create the transform
    transform = CustomTransform(is_train=True)

    flip_image, flip_mask = transform(original_image, original_mask)

    return flip_image, flip_mask

source_folder = ''
target_folder = ''

test_folder_name = 'train_image'
mask_folder_name = 'train_mask'

source_test_folder = os.path.join(source_folder, test_folder_name)
source_mask_folder = os.path.join(source_folder, mask_folder_name)
target_test_folder = os.path.join(target_folder, test_folder_name)
target_mask_folder = os.path.join(target_folder, mask_folder_name)

if not os.path.exists(target_test_folder):
    os.makedirs(target_test_folder)
if not os.path.exists(target_mask_folder):
    os.makedirs(target_mask_folder)

i = 5000
for file_name in os.listdir(source_test_folder):
    # 检查文件是否以.jpg结尾
    source_test_file = os.path.join(source_test_folder, file_name)
    print(file_name)       
    
    corresponding_name = os.path.splitext(file_name)[0] + ".png"
    source_mask_file = os.path.join(source_mask_folder, corresponding_name)

    flip_image, flip_mask = visualize_augmentation(source_test_file, source_mask_file)

    target_test_file = os.path.join(target_test_folder, f'{i}.jpg')
    target_mask_file = os.path.join(target_mask_folder, f'{i}.png')
    i+=1

    # 保存图像为PNG文件
    flip_image.save(target_test_file)
    save_aug(flip_mask, target_mask_file)

