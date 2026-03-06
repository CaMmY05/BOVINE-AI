# 🐄 BOVINE-AI: Indian Cattle Breed Recognition System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Accuracy 97.41%](https://img.shields.io/badge/Accuracy-97.41%25-brightgreen.svg)](#model-performance)
[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#license)

> **A two-stage deep learning system** for detecting and classifying Indian cattle breeds using YOLOv8 object detection + EfficientNet-B0 classification with custom head fine-tuning, achieving **97.41% average accuracy** across 6 breeds (3 cow + 3 buffalo).

---

## 📊 Results at a Glance

| Model | Architecture | Animal | Accuracy | F1 (Macro) | MCC |
|-------|-------------|--------|----------|------------|------|
| **Cow V2** | EfficientNet-B0 (timm) + Custom Head | Cow | **98.85%** | **0.984** | **0.981** |
| **Buffalo V1** | EfficientNet-B0 (timm) + Custom Head | Buffalo | **95.96%** | **0.955** | **0.937** |
| ResNet18 Cow | ResNet18 (torchvision) | Cow | 87.28% | 0.833 | — |
| Baseline V1 | EfficientNet-B0 (small dataset) | Cow | 75.65% | ~0.72 | — |

### Supported Breeds

| Cow Breeds | Accuracy | Buffalo Breeds | Accuracy |
|-----------|----------|----------------|----------|
| **Gir** | 99.72% | **Jaffarabadi** | 100.00% |
| **Sahiwal** | 99.31% | **Murrah** | 97.83% |
| **Red Sindhi** | 95.60% | **Mehsana** | 87.50% |

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────────┐
│  Input      │     │  Stage 1:        │     │  Stage 2:                   │
│  Image      │────▶│  YOLOv8n         │────▶│  EfficientNet-B0            │
│  (any size) │     │  Detection       │     │  + Custom Classification    │
│             │     │  (COCO class 19) │     │  Head (3 classes)           │
└─────────────┘     └──────────────────┘     └─────────────────────────────┘
                           │                            │
                    ┌──────▼──────┐              ┌──────▼──────┐
                    │ Bounding    │              │ Top-3 Breed │
                    │ Boxes + ROI │              │ Predictions │
                    │ Extraction  │              │ + Confidence│
                    └─────────────┘              └─────────────┘
```

### Key Innovation: Custom Classification Head

The system uses **transfer learning** with EfficientNet-B0 (pre-trained on ImageNet via `timm`) where the final fully-connected layer is replaced with a custom 3-class head fine-tuned specifically for Indian cattle breeds. Combined with:

- **Label Smoothing** (0.1) — prevents overconfident predictions
- **Class-Weighted Loss** — handles imbalanced breed distributions
- **AdamW Optimizer** with ReduceLROnPlateau scheduling
- **Strong Data Augmentation** — RandomResizedCrop, flip, rotation, color jitter

A custom from-scratch EfficientNet implementation (`models/custom_efficientnet.py`) with MBConv blocks and Squeeze-and-Excitation is also provided for reference.

---

## 🚀 Quick Start — Clone & Run

### Prerequisites

| Requirement | Version | Download |
|------------|---------|----------|
| Python | 3.11.x | [python.org](https://www.python.org/downloads/release/python-3119/) |
| NVIDIA GPU | Any with ≥4GB VRAM | — |
| CUDA Toolkit | 12.4 | [nvidia.com](https://developer.nvidia.com/cuda-12-4-0-download-archive) |
| Visual C++ Build Tools | Latest | [visualstudio.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/CaMmY05/BOVINE-AI.git
cd BOVINE-AI

# 2. Run automatic setup (creates venv + installs everything)
setup.bat

# OR manual setup:
py -3.11 -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Run the Web App

```bash
venv\Scripts\activate
streamlit run app.py
# Open http://localhost:8501
```

### Run Inference from Command Line

```python
from scripts.inference import CattleBreedPredictor

predictor = CattleBreedPredictor(
    detection_model_path='yolov8n.pt',
    classification_model_path='models/classification/cow_classifier_v2/best_model.pth',
    classes_path='models/classification/cow_classifier_v2/classes.json',
    model_arch='efficientnet_b0'
)

# Predict on an image
results = predictor.predict('path/to/cow_image.jpg', visualize=True)
```

### Regenerate Charts

```bash
venv\Scripts\activate
python -X utf8 scripts/generate_charts.py
# Charts saved to charts/ directory
```

---

## 📈 Comparison Charts

### Overall Accuracy Comparison

![Overall Accuracy](charts/01_overall_accuracy_comparison.png)

### Per-Breed Accuracy — Cow Classification

![Per-Breed Accuracy](charts/02_per_breed_cow_accuracy.png)

### F1 Score Comparison

![F1 Scores](charts/03_f1_score_comparison.png)

### Precision / Recall / F1-Score Heatmaps

![PR-F1 Heatmaps](charts/04_precision_recall_f1_heatmap.png)

### Confusion Matrices

![Confusion Matrices](charts/05_confusion_matrices.png)

### Training Curves

![Training Curves](charts/06_training_curves.png)

### Model Architecture Comparison Table

![Architecture Comparison](charts/07_model_architecture_comparison.png)

### Accuracy Improvement Journey

![Improvement Journey](charts/08_improvement_journey.png)

---

## 📁 Project Structure

```
BOVINE-AI/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── setup.bat                           # One-click environment setup
├── README.md                           # This file
├── STEP.md                             # Detailed step-by-step setup guide
├── TECH_STACK.md                       # Complete technology stack reference
├── yolov8n.pt                          # YOLOv8 nano detection model
│
├── models/
│   ├── custom_efficientnet.py          # From-scratch EfficientNet (MBConv + SE blocks)
│   ├── custom_resnet.py                # Custom ResNet32 implementation
│   └── classification/
│       ├── cow_classifier_v2/          # Best cow model (98.85%)
│       │   ├── best_model.pth          # Trained weights (48MB)
│       │   ├── classes.json            # {gir: 0, red_sindhi: 1, sahiwal: 2}
│       │   └── history.json            # Training history (20 epochs)
│       ├── buffalo_classifier_v1/      # Best buffalo model (95.96%)
│       │   ├── best_model.pth          # Trained weights (48MB)
│       │   ├── classes.json            # {jaffarabadi: 0, mehsana: 1, murrah: 2}
│       │   └── history.json            # Training history (28 epochs)
│       ├── resnet18_cow_v1/            # ResNet18 comparison (87.28%)
│       ├── resnet18_buffalo_v1/        # ResNet18 buffalo comparison
│       ├── resnet34_cow_v1/            # ResNet34 comparison
│       └── resnet34_buffalo_v1/        # ResNet34 buffalo comparison
│
├── scripts/
│   ├── inference.py                    # CattleBreedPredictor (detection + classification)
│   ├── generate_charts.py              # Generates all comparison charts
│   ├── train_cow_classifier_v2.py      # Cow model training script
│   ├── train_buffalo_classifier.py     # Buffalo model training script
│   ├── evaluate_v2.py                  # Cow model evaluation
│   ├── evaluate_buffalo_model.py       # Buffalo model evaluation
│   ├── multi_view_analysis.py          # Three-view (left/front/right) analysis
│   ├── dataset.py                      # PyTorch Dataset class
│   ├── model_registry.py              # Model registry utilities
│   └── data_collection/               # Data acquisition scripts
│       ├── download_roboflow_datasets.py       # Roboflow API dataset download
│       ├── download_all_research_datasets.py   # Academic research datasets
│       ├── download_buffalo_images.py          # Buffalo-specific image scraper
│       ├── download_google_simple.py           # Google Images scraper
│       ├── download_images_bulk.py             # Bulk image downloader
│       ├── download_kaggle_datasets.py         # Kaggle dataset download
│       ├── download_sample_data.py             # Sample data fetcher
│       ├── download_auto.py                    # Automated download pipeline
│       ├── organize_all_data_and_download_buffalo.py  # Data org + download
│       └── process_all_downloads.py            # Post-download processing
│
├── data/processed_v2/                  # Pre-processed datasets
│   ├── cows/{train,val,test}/          # 6,788 images (Gir, Red Sindhi, Sahiwal)
│   └── buffaloes/{train,val,test}/     # 686 images (Jaffarabadi, Mehsana, Murrah)
│
├── results/
│   ├── evaluation_v2/                  # Cow evaluation (confusion matrix, metrics)
│   └── buffalo_evaluation/             # Buffalo evaluation (confusion matrix, metrics)
│
├── charts/                             # Generated comparison charts (8 PNGs)
│
└── docs/
    ├── FINAL_MVP_SUMMARY.md            # Comprehensive project summary
    ├── MODEL_ARCHITECTURE_AND_ANALYSIS.md
    └── PROJECT_SUMMARY.md
```

---

## 🔬 Detailed Metrics

### Cow Classifier V2 — EfficientNet-B0

```
              precision    recall  f1-score   support

         gir      0.989     0.997     0.993       357
  red_sindhi      0.974     0.956     0.965       159
     sahiwal      0.993     0.993     0.993       437

    accuracy                          0.988       953
   macro avg      0.985     0.982     0.984       953
weighted avg      0.988     0.988     0.988       953

Matthews Correlation Coefficient (MCC): 0.9814
Cohen's Kappa: 0.9814
Top-3 Accuracy: 100.00%
```

### Buffalo Classifier V1 — EfficientNet-B0

```
              precision    recall  f1-score   support

 jaffarabadi      0.967     1.000     0.983        29
     mehsana      0.955     0.875     0.913        24
      murrah      0.957     0.978     0.968        46

    accuracy                          0.960        99
   macro avg      0.960     0.951     0.955        99
weighted avg      0.959     0.960     0.959        99

Matthews Correlation Coefficient (MCC): 0.9370
Cohen's Kappa: 0.9365
Top-3 Accuracy: 100.00%
```

### Improvement Over Baseline (Cow Breeds)

| Breed | V1 Baseline | ResNet18 | V2 EfficientNet | Improvement |
|-------|-------------|----------|-----------------|-------------|
| Gir | 91.11% | 93.16% | **99.72%** | +8.61% |
| Sahiwal | 80.00% | 93.64% | **99.31%** | +19.31% |
| Red Sindhi | 30.00% | 56.47% | **95.60%** | **+65.60%** |
| **Overall** | **75.65%** | **87.28%** | **98.85%** | **+23.20%** |

---

## 🛠️ Technology Stack

### Languages
| Language | Usage |
|----------|-------|
| **Python 3.11** | Core application, ML training, inference |
| **HTML/CSS/JS** | Streamlit auto-generated web UI |
| **JSON** | Model configs, class mappings, metrics |
| **Markdown** | Documentation |
| **Batch Script** | Windows setup automation |

### Libraries & Frameworks
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.6.0+cu124 | Deep learning framework |
| **torchvision** | 0.21.0 | Image transforms, pretrained models |
| **timm** | 1.0.25 | EfficientNet-B0 model zoo |
| **Ultralytics** | 8.4.21 | YOLOv8 object detection |
| **Streamlit** | 1.55.0 | Interactive web application |
| **OpenCV** | 4.11.0 | Image processing, visualization |
| **scikit-learn** | 1.8.0 | Evaluation metrics, confusion matrix |
| **matplotlib** | 3.x | Chart generation |
| **seaborn** | 0.13+ | Statistical visualizations |
| **NumPy** | 1.26.4 | Numerical computing |
| **Pandas** | 2.x | Data analysis |
| **Pillow** | 12.0 | Image loading |
| **albumentations** | 1.3+ | Data augmentation |
| **tqdm** | 4.66+ | Progress bars |
| **TensorBoard** | 2.15+ | Training visualization |

### Hardware (Developed On)
| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 4000 Ada Generation (12GB VRAM) |
| **CUDA** | 12.4 |
| **Driver** | 582.16 |
| **OS** | Windows 11 |

---

## 📚 Citations & References

### Model Architectures

1. **EfficientNet** — Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. Proceedings of the 36th International Conference on Machine Learning (ICML 2019). [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

2. **ResNet** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

3. **YOLOv8** — Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. Ultralytics. [GitHub](https://github.com/ultralytics/ultralytics)

4. **Squeeze-and-Excitation Networks** — Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018). [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)

### Libraries

5. **timm (PyTorch Image Models)** — Wightman, R. (2019). *PyTorch Image Models*. [GitHub](https://github.com/huggingface/pytorch-image-models). DOI: [10.5281/zenodo.4414861](https://doi.org/10.5281/zenodo.4414861)

6. **PyTorch** — Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems 32 (NeurIPS 2019). [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

7. **Streamlit** — Streamlit Inc. (2019). *Streamlit — The fastest way to build data apps*. [streamlit.io](https://streamlit.io/)

### Dataset

8. **Indian Bovine Recognition Dataset** — Sourced via Roboflow. 15,077 images across 41 Indian cattle breeds. Used under the dataset's license terms for academic research.

### Techniques

9. **Transfer Learning** — Zhuang, F., et al. (2020). *A Comprehensive Survey on Transfer Learning*. Proceedings of the IEEE, 109(1), 43-76. [arXiv:1911.02685](https://arxiv.org/abs/1911.02685)

10. **Label Smoothing** — Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)

11. **AdamW Optimizer** — Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

---

## 📝 Training Configuration

### Best Cow Model (V2)
```python
Architecture:     EfficientNet-B0 (timm, pretrained on ImageNet)
Custom Head:      FC(1280 → 3)  # 3 cow breeds
Optimizer:        AdamW (lr=0.001, weight_decay=0.01)
Loss:             CrossEntropyLoss + Label Smoothing (0.1) + Class Weights
Scheduler:        ReduceLROnPlateau (patience=5, factor=0.5)
Early Stopping:   10 epochs patience
Batch Size:       32
Image Size:       224 x 224
Augmentation:     RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter
Dataset:          6,788 images (70/15/15 split)
Epochs:           50 planned, early stopped at 20
Final Test Acc:   98.85%
```

### Best Buffalo Model (V1)
```python
Architecture:     EfficientNet-B0 (timm, pretrained on ImageNet)
Custom Head:      FC(1280 → 3)  # 3 buffalo breeds
Optimizer:        AdamW (lr=0.001, weight_decay=0.01)
Loss:             CrossEntropyLoss + Label Smoothing (0.1) + Class Weights
Scheduler:        ReduceLROnPlateau (patience=5, factor=0.5)
Early Stopping:   10 epochs patience
Batch Size:       32
Image Size:       224 x 224
Augmentation:     RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter
Dataset:          686 images (70/15/15 split)
Epochs:           30 planned, early stopped at 28
Final Test Acc:   95.96%
```

---

## 📄 License

This project is developed for **academic and research purposes**. 

- **Original Code & Trained Models**: Copyrighted by the author
- **EfficientNet/ResNet Architectures**: Published research (see citations)
- **YOLOv8**: AGPL-3.0 License (Ultralytics)
- **timm**: Apache 2.0 License
- **Dataset**: Subject to Roboflow dataset license terms

---

## 🤝 Acknowledgments

- **Roboflow** — Indian Bovine Recognition Dataset (15,077 images, 41 breeds)
- **Google Brain** — EfficientNet architecture
- **Microsoft Research** — ResNet architecture
- **Ultralytics** — YOLOv8 detection framework
- **Ross Wightman** — timm (PyTorch Image Models) library
