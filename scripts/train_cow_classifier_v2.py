"""
Train Cow Breed Classifier - Version 2
With proper epoch configuration to avoid overfitting/underfitting
Preserves base model and creates new model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import json
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
        
        # Load all images
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

class BreedClassifier:
    def __init__(self, num_classes, model_name='efficientnet_b0', class_weights=None):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Build model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Loss and optimizer
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"Using class weights: {class_weights}")
        
        # Label smoothing helps prevent overfitting
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def build_model(self):
        # Use EfficientNet-B0 (good balance of speed and accuracy)
        model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        return model
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Early stopping patience: 10 epochs")
        print(f"Learning rate reduction patience: 5 epochs")
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_epoch = 0
        patience_counter = 0
        early_stop_patience = 10  # Stop if no improvement for 10 epochs
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'classes': train_loader.dataset.classes
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs (best: {self.best_val_acc:.2f}% at epoch {best_epoch})")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {best_epoch}")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        
        # Save history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.2f}%")

def calculate_optimal_epochs(dataset_size):
    """
    Calculate optimal number of epochs based on dataset size
    to avoid overfitting and underfitting
    """
    if dataset_size < 500:
        # Small dataset: risk of overfitting, use fewer epochs
        return 20
    elif dataset_size < 1000:
        # Medium dataset: moderate epochs
        return 30
    elif dataset_size < 2000:
        # Large dataset: can train longer
        return 40
    else:
        # Very large dataset: train even longer
        return 50

def main():
    print("="*60)
    print("COW BREED CLASSIFIER TRAINING - V2")
    print("="*60)
    
    # Paths
    DATA_DIR = "data/processed_v2/cows/"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    
    # Check if data exists
    if not os.path.exists(TRAIN_DIR):
        print(f"✗ Training data not found at: {TRAIN_DIR}")
        print("Run: python scripts/prepare_data_v2.py first")
        return
    
    # Data transforms with moderate augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Moderate cropping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Moderate rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BreedDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = BreedDataset(VAL_DIR, transform=val_transform)
    
    print(f"\ntrain dataset: {len(train_dataset)} images, {train_dataset.num_classes} classes")
    print(f"val dataset: {len(val_dataset)} images, {val_dataset.num_classes} classes")
    
    # Calculate optimal epochs
    optimal_epochs = calculate_optimal_epochs(len(train_dataset))
    print(f"\n📊 Dataset size: {len(train_dataset)} images")
    print(f"🎯 Optimal epochs: {optimal_epochs}")
    print(f"   (Calculated to avoid overfitting/underfitting)")
    
    # Calculate class weights
    class_counts = {}
    for _, label in train_dataset:
        label_idx = label.item() if torch.is_tensor(label) else label
        class_counts[label_idx] = class_counts.get(label_idx, 0) + 1
    
    total_samples = len(train_dataset)
    class_weights = torch.zeros(train_dataset.num_classes)
    for class_idx in range(train_dataset.num_classes):
        count = class_counts.get(class_idx, 1)
        class_weights[class_idx] = total_samples / (train_dataset.num_classes * count)
    
    print(f"\nClass distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize classifier
    classifier = BreedClassifier(
        num_classes=train_dataset.num_classes,
        model_name='efficientnet_b0',
        class_weights=class_weights
    )
    
    # Save directory (preserve base model)
    SAVE_DIR = "models/classification/cow_classifier_v2/"
    print(f"\nModel will be saved to: {SAVE_DIR}")
    print(f"Base model (v1) preserved at: models/classification/breed_classifier_v1/")
    
    # Train
    classifier.train(train_loader, val_loader, num_epochs=optimal_epochs, save_dir=SAVE_DIR)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {SAVE_DIR}best_model.pth")
    print(f"Best validation accuracy: {classifier.best_val_acc:.2f}%")
    print("\nNext steps:")
    print("1. Evaluate: python scripts/evaluate_v2.py")
    print("2. Test with Streamlit: streamlit run app.py")

if __name__ == "__main__":
    main()
