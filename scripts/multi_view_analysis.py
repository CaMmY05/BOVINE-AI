import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

class ThreeViewAnalyzer:
    """
    Divides cattle image into three regions: Left, Front, Right
    Extracts features from each region
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def divide_into_views(self, image):
        """
        Divide image into three vertical regions
        Returns: dict with 'left', 'front', 'right' PIL images
        """
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        width, height = image.size
        third = width // 3
        
        views = {
            'left': image.crop((0, 0, third, height)),
            'front': image.crop((third, 0, 2*third, height)),
            'right': image.crop((2*third, 0, width, height))
        }
        
        return views
    
    def preprocess_views(self, image_path):
        """
        Preprocess image and return tensors for each view
        Returns: dict with 'left', 'front', 'right' tensors
        """
        
        views = self.divide_into_views(image_path)
        
        processed_views = {}
        for view_name, view_img in views.items():
            processed_views[view_name] = self.transform(view_img)
        
        return processed_views
    
    def visualize_views(self, image_path, output_path=None):
        """
        Visualize the three views side by side
        """
        import matplotlib.pyplot as plt
        
        views = self.divide_into_views(image_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (view_name, view_img) in enumerate(views.items()):
            axes[idx].imshow(view_img)
            axes[idx].set_title(f'{view_name.capitalize()} View')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

# Test
if __name__ == "__main__":
    analyzer = ThreeViewAnalyzer()
    
    # Test on a sample image
    test_image = "data/processed/train/roi_images/train_00001.jpg"
    
    if os.path.exists(test_image):
        analyzer.visualize_views(test_image, "results/three_view_demo.png")
        print("Three-view visualization saved!")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run data preparation and ROI extraction first.")
