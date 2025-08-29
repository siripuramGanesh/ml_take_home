from typing import List

import cv2
import numpy as np


class ImageAugmenter:
    """Data augmentation using OpenCV and custom transformations"""
    
    @staticmethod
    def augment_image(image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple augmentations to an image"""
        augmented_images = [image]
        
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))
        
        # Brightness adjustments
        for alpha in [0.8, 1.2]:
            bright_image = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            augmented_images.append(bright_image)
        
        # Gaussian blur
        blur_image = cv2.GaussianBlur(image, (5, 5), 0)
        augmented_images.append(blur_image)
        
        return augmented_images
    
    @staticmethod
    def random_augment(image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to an image"""
        augmented = image.copy()
        
        # Random flip
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # Random brightness
        brightness = np.random.uniform(0.7, 1.3)
        augmented = np.clip(augmented.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast
        contrast = np.random.uniform(0.7, 1.3)
        augmented = np.clip(128 + contrast * (augmented.astype(np.float32) - 128), 0, 255).astype(np.uint8)
        
        return augmented