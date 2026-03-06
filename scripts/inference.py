import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import sys
from ultralytics import YOLO
import cv2
import numpy as np
import timm

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CattleBreedPredictor:
    def __init__(self, 
                 detection_model_path='yolov8n.pt',
                 classification_model_path='models/classification/breed_classifier_v1/best_model.pth',
                 classes_path='data/processed/classes.json',
                 model_arch='efficientnet_b0'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_arch = model_arch.lower()
        
        # Load detection model
        self.detector = YOLO(detection_model_path)
        
        # Load classification model
        with open(classes_path, 'r') as f:
            class_data = json.load(f)
        
        # Handle both list and dict formats for class_to_idx
        if isinstance(class_data, list):
            self.idx_to_class = {i: class_name for i, class_name in enumerate(class_data)}
            class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        elif isinstance(class_data, dict):
            class_to_idx = class_data
            self.idx_to_class = {v: k for k, v in class_data.items()}
        else:
            raise ValueError("classes.json must contain either a list or a dictionary")
        
        num_classes = len(class_to_idx)
        
        # Initialize model based on architecture
        self.model = self._initialize_model(num_classes)
        self._load_model_weights(classification_model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _initialize_model(self, num_classes):
        """Initialize the model architecture"""
        if self.model_arch == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            print("Using ResNet18 architecture")
        elif self.model_arch == 'resnet32':
            # Using ResNet34 as ResNet32 since we don't have a proper ResNet32 implementation
            model = models.resnet34(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            print("Using ResNet34 as ResNet32 architecture")
        elif self.model_arch == 'efficientnet_b0':
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
            print("Using EfficientNet-B0 architecture")
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_arch}")
        return model
    
    def _load_model_weights(self, model_path):
        """Load model weights with appropriate handling for different architectures"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Handle DataParallel or other parallel wrappers
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Special handling for ResNet models
            if self.model_arch in ['resnet18', 'resnet32']:
                # Remove 'fc.' prefix if present in state dict
                state_dict = {k.replace('fc.', '') if k.startswith('fc.') else k: v 
                            for k, v in state_dict.items()}
                
                # For ResNet32 (which is actually ResNet34), we need to handle the layer names
                if self.model_arch == 'resnet32':
                    # Map layer names from ResNet32 to ResNet34
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('layer'):
                            # Map layer blocks (e.g., layer1.0 -> layer1.0)
                            parts = k.split('.')
                            if len(parts) > 1 and parts[0].startswith('layer'):
                                layer_num = int(parts[0][5:])
                                # ResNet34 has more layers than ResNet32, but we'll map them directly
                                # and let the rest be handled by strict=False
                                new_k = k
                                new_state_dict[new_k] = v
                                continue
                        new_state_dict[k] = v
                    state_dict = new_state_dict
            
            # Load state dict with strict=False to allow partial loading
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded weights from {model_path}")
            
            # Print which layers were loaded successfully
            model_dict = self.model.state_dict()
            loaded_layers = [k for k in state_dict.keys() if k in model_dict]
            print(f"Successfully loaded layers: {len(loaded_layers)}/{len(state_dict)}")
            
            # Print missing keys if any
            missing_keys = [k for k in state_dict.keys() if k not in model_dict]
            if missing_keys:
                print(f"Warning: The following keys were not found in the model: {missing_keys}")
            
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Attempting to load with strict=False...")
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded with strict=False")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model weights even with strict=False: {e2}")
        
        # Transform for classification
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
        print(f"Classes: {list(self.idx_to_class.values())}")
    
    def detect_and_extract_roi(self, image_path, confidence_threshold=0.4):
        """
        Detect cattle and extract ROI
        """
        results = self.detector.predict(
            image_path,
            classes=[19],  # COCO cow class
            conf=confidence_threshold,
            verbose=False
        )
        
        img = cv2.imread(str(image_path))
        rois = []
        boxes_info = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Add padding
                h, w = y2 - y1, x2 - x1
                padding_h, padding_w = int(h * 0.1), int(w * 0.1)
                
                y1 = max(0, y1 - padding_h)
                y2 = min(img.shape[0], y2 + padding_h)
                x1 = max(0, x1 - padding_w)
                x2 = min(img.shape[1], x2 + padding_w)
                
                roi = img[y1:y2, x1:x2]
                
                if roi.size > 0:
                    rois.append(roi)
                    boxes_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        return rois, boxes_info
    
    def classify_breed(self, roi_image, top_k=3):
        """
        Classify breed from ROI
        Returns top-k predictions with scores
        """
        # Convert to PIL
        if isinstance(roi_image, np.ndarray):
            roi_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        
        # Transform and predict
        img_tensor = self.transform(roi_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'breed': self.idx_to_class[idx.item()],
                    'confidence': prob.item(),
                    'score': prob.item() * 100
                })
        
        return predictions
    
    def predict(self, image_path, visualize=False):
        """
        Complete pipeline: detect -> extract -> classify
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        # Step 1: Detection
        print("\n[1/3] Detecting cattle...")
        rois, boxes_info = self.detect_and_extract_roi(image_path)
        
        if not rois:
            print("❌ No cattle detected in the image!")
            return None
        
        print(f"✓ Detected {len(rois)} cattle")
        
        # Step 2: Classification for each ROI
        print("\n[2/3] Classifying breeds...")
        all_predictions = []
        
        for idx, roi in enumerate(rois):
            print(f"\n  Cattle #{idx+1}:")
            predictions = self.classify_breed(roi, top_k=3)
            
            for rank, pred in enumerate(predictions, 1):
                print(f"    {rank}. {pred['breed']:20s} - {pred['score']:.2f}% confidence")
            
            all_predictions.append({
                'roi_index': idx,
                'bbox': boxes_info[idx]['bbox'],
                'detection_confidence': boxes_info[idx]['confidence'],
                'breed_predictions': predictions
            })
        
        # Step 3: Visualization
        if visualize:
            print("\n[3/3] Creating visualization...")
            output_path = self.visualize_predictions(image_path, all_predictions)
            print(f"✓ Saved visualization to: {output_path}")
        
        return all_predictions
    
    def visualize_predictions(self, image_path, predictions, output_dir='results/predictions'):
        """
        Draw bounding boxes and predictions on image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(str(image_path))
        
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            top_breed = pred['breed_predictions'][0]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"{top_breed['breed']}: {top_breed['score']:.1f}%"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                img, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1),
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
            )
        
        # Save
        output_filename = os.path.basename(image_path).replace('.', '_predicted.')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        return output_path
    
    def predict_batch(self, image_dir, output_csv='results/predictions.csv'):
        """
        Batch prediction on directory of images
        """
        import pandas as pd
        from pathlib import Path
        
        image_files = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.jpeg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        results = []
        
        for img_path in image_files:
            predictions = self.predict(str(img_path), visualize=True)
            
            if predictions:
                for pred in predictions:
                    top_breed = pred['breed_predictions'][0]
                    results.append({
                        'image': img_path.name,
                        'roi_index': pred['roi_index'],
                        'predicted_breed': top_breed['breed'],
                        'confidence': top_breed['score'],
                        'top2_breed': pred['breed_predictions'][1]['breed'] if len(pred['breed_predictions']) > 1 else '',
                        'top2_confidence': pred['breed_predictions'][1]['score'] if len(pred['breed_predictions']) > 1 else 0
                    })
        
        # Save to CSV
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = CattleBreedPredictor(
        detection_model_path='yolov8n.pt',
        classification_model_path='models/classification/breed_classifier_v1/best_model.pth',
        classes_path='data/processed/classes.json'
    )
    
    # Single image prediction
    test_image = 'test_images/cattle_1.jpg'
    
    if os.path.exists(test_image):
        predictions = predictor.predict(test_image, visualize=True)
    else:
        print(f"Test image not found: {test_image}")
        print("Please add test images to the test_images/ directory")
    
    # Batch prediction (uncomment to use)
    # predictor.predict_batch('test_images/', output_csv='results/batch_predictions.csv')
