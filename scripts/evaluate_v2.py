"""
Evaluate Cow Breed Classifier V2
Complete evaluation with metrics and visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

class BreedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = Path(data_dir) / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
        
        self.num_classes = len(self.classes)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_model(model_path, num_classes):
    """Load trained model"""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Cow Breed Classifier V2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")

def main():
    print("="*60)
    print("COW BREED CLASSIFIER V2 - EVALUATION")
    print("="*60)
    
    # Paths
    MODEL_PATH = "models/classification/cow_classifier_v2/best_model.pth"
    DATA_DIR = "data/processed_v2/cows/test"
    RESULTS_DIR = "results/evaluation_v2"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found at: {MODEL_PATH}")
        print("Make sure training completed successfully.")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = BreedDataset(DATA_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"test dataset: {len(test_dataset)} images, {test_dataset.num_classes} classes")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, test_dataset.num_classes)
    model = model.to(device)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device, test_dataset.classes)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(labels, predictions, target_names=test_dataset.classes, digits=3)
    print(report)
    
    # Overall accuracy
    accuracy = (predictions == labels).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for idx, class_name in enumerate(test_dataset.classes):
        class_mask = labels == idx
        class_acc = (predictions[class_mask] == labels[class_mask]).mean() * 100
        class_count = class_mask.sum()
        print(f"  {class_name:15s}: {class_acc:5.2f}% ({class_count} images)")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, test_dataset.classes, cm_path)
    
    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0
    for i in range(len(labels)):
        top3_preds = np.argsort(probabilities[i])[-3:]
        top5_preds = np.argsort(probabilities[i])[-5:]
        if labels[i] in top3_preds:
            top3_correct += 1
        if labels[i] in top5_preds:
            top5_correct += 1
    
    top3_acc = top3_correct / len(labels) * 100
    top5_acc = top5_correct / len(labels) * 100
    
    print(f"\nTop-1 Accuracy: {accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    if test_dataset.num_classes >= 5:
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    # Find misclassifications
    print("\n" + "="*60)
    print("TOP 10 CONFIDENT MISCLASSIFICATIONS")
    print("="*60)
    
    misclassified = predictions != labels
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) > 0:
        misclassified_probs = np.max(probabilities[misclassified_indices], axis=1)
        top_misclassified = misclassified_indices[np.argsort(misclassified_probs)[-10:][::-1]]
        
        for rank, idx in enumerate(top_misclassified, 1):
            true_label = test_dataset.classes[labels[idx]]
            pred_label = test_dataset.classes[predictions[idx]]
            confidence = np.max(probabilities[idx]) * 100
            print(f"\n{rank}. Image index: {idx}")
            print(f"   True: {true_label:15s} | Predicted: {pred_label:15s} | Confidence: {confidence:.2f}%")
    
    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        'per_class_accuracy': {
            class_name: float((predictions[labels == idx] == labels[labels == idx]).mean() * 100)
            for idx, class_name in enumerate(test_dataset.classes)
        },
        'top3_accuracy': float(top3_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'num_test_images': len(test_dataset),
        'classes': test_dataset.classes
    }
    
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"  - Confusion matrix: {cm_path}")
    print(f"  - Evaluation results: {results_path}")
    
    # Compare with base model
    print("\n" + "="*60)
    print("COMPARISON WITH BASE MODEL")
    print("="*60)
    print("Base Model (V1):")
    print("  Overall: 75.65%")
    print("  Gir: 91.11%, Sahiwal: 80.00%, Red Sindhi: 30.00%")
    print()
    print(f"New Model (V2):")
    print(f"  Overall: {accuracy:.2f}%")
    for idx, class_name in enumerate(test_dataset.classes):
        class_mask = labels == idx
        class_acc = (predictions[class_mask] == labels[class_mask]).mean() * 100
        print(f"  {class_name.capitalize()}: {class_acc:.2f}%")
    print()
    print(f"Improvement: {accuracy - 75.65:+.2f}%")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Test with Streamlit:")
    print("   streamlit run app.py")
    print()
    print("2. If satisfied, proceed with buffalo breeds:")
    print("   python scripts\\download_buffalo_images.py")

if __name__ == "__main__":
    main()
