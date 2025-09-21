import os
import logging

import cv2
import albumentations as A

logger = logging.getLogger(__name__)

def augment_images(image_dir):
    """Apply augmentations to images in the specified directory."""
    transform = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])
    
    augmented_images = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)["image"]
        augmented_images.append({
            "image": augmented,
            "image_id": img_name,
            "original_path": img_path
        })
        logger.info(f"Augmented image: {img_name}")
    
    return augmented_images