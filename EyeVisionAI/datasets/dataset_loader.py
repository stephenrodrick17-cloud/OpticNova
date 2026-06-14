import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class GlaucomaClassificationDataset(Dataset):
    """
    Dataset for Classification.
    Assumes a folder structure where images are in class-specific subfolders.
    e.g., data_dir/glaucoma/img1.jpg, data_dir/normal/img2.jpg
    """
    def __init__(self, data_dir: str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_paths = []
        self.labels = []
        
        # Typically class 0 = normal, class 1 = glaucoma
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.float32)


class FundusSegmentationDataset(Dataset):
    """
    Dataset for Segmentation.
    Assumes a folder structure:
    data_dir/
      images/
         img1.jpg
      masks/
         img1.png (or .jpg, matching name)
    """
    def __init__(self, data_dir: str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'mask')
        
        # Use glob to find all images
        self.image_paths = sorted(glob(os.path.join(self.images_dir, '*.*')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_filename = os.path.basename(img_path)
        
        # Assume mask has same basename (might have different extension, handle appropriately)
        base_name, _ = os.path.splitext(img_filename)
        mask_path_png = os.path.join(self.masks_dir, base_name + '.png')
        mask_path_jpg = os.path.join(self.masks_dir, base_name + '.jpg')
        mask_path_bmp = os.path.join(self.masks_dir, base_name + '.bmp')
        mask_path_tif = os.path.join(self.masks_dir, base_name + '.tif')
        
        if os.path.exists(mask_path_png):
            mask_path = mask_path_png
        elif os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.exists(mask_path_bmp):
            mask_path = mask_path_bmp
        elif os.path.exists(mask_path_tif):
            mask_path = mask_path_tif
        else:
            raise FileNotFoundError(f"No mask found for {img_filename} in {self.masks_dir}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # REFUGE Masks: 0=Cup, 128=Disc Rim, 255=Background
        # Disc = Cup + Rim (values 0 and 128)
        disc_mask = (mask < 200).astype(np.float32)
        # Cup = only 0
        cup_mask = (mask < 50).astype(np.float32)
        
        if self.transforms:
            # Albumentations expects mask to be HxW or HxWxC
            mask_combined = np.stack([disc_mask, cup_mask], axis=-1)
            augmented = self.transforms(image=image, mask=mask_combined)
            image = augmented['image']
            mask_combined = augmented['mask']
            
            # Convert to PyTorch Tensor (C, H, W)
            mask_tensor = torch.tensor(mask_combined).permute(2, 0, 1)
        else:
            mask_tensor = torch.stack([torch.tensor(disc_mask), torch.tensor(cup_mask)], dim=0)
            
        return image, mask_tensor
