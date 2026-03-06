import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CattleBreedDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, use_three_views=False):
        """
        Args:
            root_dir: data/processed/
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
            use_three_views: If True, returns three views per image
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_three_views = use_three_views
        
        # Load class mapping
        classes_file = os.path.join(root_dir, 'classes.json')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_to_idx = json.load(f)
        else:
            print(f"Warning: {classes_file} not found. Creating default mapping.")
            self.class_to_idx = {}
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Try roi_images first, fallback to images
        images_dir = os.path.join(root_dir, split, 'roi_images')
        if not os.path.exists(images_dir):
            images_dir = os.path.join(root_dir, split, 'images')
        
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(images_dir, img_file)
                    
                    # Get label from corresponding label file
                    label_file = os.path.join(root_dir, split, 'labels', 
                                             img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                    
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            label = int(f.read().strip())
                        
                        self.images.append(img_path)
                        self.labels.append(label)
        
        print(f"{split} dataset: {len(self.images)} images, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.use_three_views:
            # Divide into three views
            from multi_view_analysis import ThreeViewAnalyzer
            analyzer = ThreeViewAnalyzer()
            views = analyzer.divide_into_views(image)
            
            if self.transform:
                views = {k: self.transform(v) for k, v in views.items()}
            
            # Stack views along a new dimension
            image_tensor = torch.stack([views['left'], views['front'], views['right']])
            
        else:
            if self.transform:
                image = self.transform(image)
            image_tensor = image
        
        return image_tensor, label
