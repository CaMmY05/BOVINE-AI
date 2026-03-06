"""
Model Registry for Cattle Breed Classification

This script provides a centralized registry of all models, their configurations,
dataset paths, and evaluation results.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import pandas as pd
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    name: str
    architecture: str
    version: str
    supported_animals: List[str]
    dataset_path: Path
    model_path: Path
    train_script: str
    evaluate_script: str
    input_size: tuple = (224, 224)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    
    def to_dict(self):
        return {
            'name': self.name,
            'architecture': self.architecture,
            'version': self.version,
            'supported_animals': self.supported_animals,
            'dataset_path': str(self.dataset_path),
            'model_path': str(self.model_path),
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }

# Model Registry
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # EfficientNet Models
    'efficientnet_b0_v1': ModelConfig(
        name='EfficientNet-B0',
        architecture='efficientnet_b0',
        version='v1',
        supported_animals=['cow'],
        dataset_path=DATA_DIR / 'processed_v2/cows',  # Original dataset
        model_path=MODELS_DIR / 'classification/cow_classifier_v2',
        train_script='train_efficientnet.py',
        evaluate_script='evaluate_efficientnet_v2.py',
        epochs=30
    ),
    'efficientnet_b0_v2': ModelConfig(
        name='EfficientNet-B0',
        architecture='efficientnet_b0',
        version='v2',
        supported_animals=['cow', 'buffalo'],
        dataset_path=DATA_DIR / 'processed_v2',
        model_path=MODELS_DIR / 'classification/cow_classifier_v2',
        train_script='train_efficientnet_v2.py',
        evaluate_script='evaluate_efficientnet_v2.py',
        epochs=50
    ),
    'efficientnet_b0_v3': ModelConfig(
        name='EfficientNet-B0',
        architecture='efficientnet_b0',
        version='v3',
        supported_animals=['cow'],
        dataset_path=DATA_DIR / 'processed_v3/cows',
        model_path=MODELS_DIR / 'classification/cow_classifier_v3',
        train_script='train_efficientnet_v3.py',
        evaluate_script='evaluate_efficientnet_v2.py',
        epochs=40
    ),
    
    # ResNet Models
    'resnet18_v1': ModelConfig(
        name='ResNet18',
        architecture='resnet18',
        version='v1',
        supported_animals=['cow', 'buffalo'],
        dataset_path=DATA_DIR / 'processed_v2',
        model_path=MODELS_DIR / 'classification/resnet18_cow_v1',
        train_script='train_resnet.py',
        evaluate_script='evaluate_model.py',
        epochs=50
    ),
    'resnet32_v1': ModelConfig(
        name='ResNet32',
        architecture='resnet34',  # Using ResNet34 as ResNet32
        version='v1',
        supported_animals=['cow', 'buffalo'],
        dataset_path=DATA_DIR / 'processed_v2',
        model_path=MODELS_DIR / 'classification/resnet34_cow_v1',
        train_script='train_resnet.py',
        evaluate_script='evaluate_model.py',
        epochs=60
    )
}

def get_model_config(model_id: str) -> ModelConfig:
    """Get model configuration by ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_id} not found in registry")
    return MODEL_REGISTRY[model_id]

def list_models() -> List[dict]:
    """List all registered models with basic info."""
    return [{
        'id': model_id,
        'name': config.name,
        'architecture': config.architecture,
        'version': config.version,
        'supported_animals': config.supported_animals,
        'model_path': str(config.model_path)
    } for model_id, config in MODEL_REGISTRY.items()]

def get_evaluation_results(model_id: str) -> dict:
    """Get evaluation results for a model."""
    config = get_model_config(model_id)
    results_file = RESULTS_DIR / f"{model_id}_evaluation.json"
    
    if not results_file.exists():
        return {"status": "not_evaluated", "message": f"No evaluation results found for {model_id}"}
    
    with open(results_file, 'r') as f:
        return json.load(f)

def save_evaluation_results(model_id: str, results: dict):
    """Save evaluation results for a model."""
    config = get_model_config(model_id)
    results_file = RESULTS_DIR / f"{model_id}_evaluation.json"
    
    # Add metadata
    results['evaluation_timestamp'] = pd.Timestamp.now().isoformat()
    results['model_id'] = model_id
    results['model_config'] = config.to_dict()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Example usage
    print("Available models:")
    for model in list_models():
        print(f"- {model['name']} {model['version']} ({model['architecture']}): {model['supported_animals']}")
        print(f"  Path: {model['model_path']}")
