import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 512):
    """
    Returns albumentations transforms for training.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.2),
        A.OneOf([
            A.HueSaturationValue(10, 15, 10),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(image_size: int = 512):
    """
    Returns albumentations transforms for validation/inference.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
