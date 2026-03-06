# 🛠️ TECHNOLOGY STACK — Complete Reference

> Every language, framework, package, and tool used in BOVINE-AI, with exact locations.

---

## Languages

| Language | Files | Purpose |
|----------|-------|---------|
| **Python 3.11** | `app.py`, `scripts/*.py`, `models/*.py` | Core application, ML training, inference, evaluation, visualization |
| **JSON** | `classes.json`, `history.json`, `metrics.json`, `evaluation_results.json`, `enhanced_metrics.json` | Model configs, class mappings, training history, evaluation metrics |
| **Markdown** | `README.md`, `STEP.md`, `docs/*.md` | Documentation, setup guide, project summaries |
| **Batch Script** | `setup.bat` | Windows environment setup automation |

---

## Deep Learning Frameworks

### PyTorch (`torch`) — v2.6.0+cu124
> Core deep learning framework powering all model operations.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | `torch.load()` for model weight loading, `torch.no_grad()` for inference, `.to('cuda')` for GPU, `torch.nn.functional.softmax()` for predictions |
| `scripts/train_cow_classifier_v2.py` | Training loop, `torch.optim.AdamW`, `nn.CrossEntropyLoss`, `lr_scheduler.ReduceLROnPlateau`, early stopping, model saving |
| `scripts/train_buffalo_classifier.py` | Same as cow training — full training pipeline with GPU acceleration |
| `models/custom_efficientnet.py` | `nn.Module`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`, `nn.AdaptiveAvgPool2d` — from-scratch EfficientNet |
| `models/custom_resnet.py` | `nn.Module`, `nn.Conv2d`, `BasicBlock` — custom ResNet32 architecture |
| `scripts/dataset.py` | `torch.utils.data.Dataset` for breed image loading |
| `app.py` | `torch.cuda.is_available()` — GPU detection for sidebar display |

### torchvision — v0.21.0+cu124
> Image preprocessing transforms and pretrained model access.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | `torchvision.transforms.Compose([Resize, CenterCrop, ToTensor, Normalize])` for inference preprocessing |
| `models/custom_resnet.py` | `torchvision.models.resnet18()` / `resnet34()` for pretrained ResNet backbones |

---

## Object Detection

### Ultralytics YOLOv8 — v8.4.21
> Real-time cattle detection in images.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | `from ultralytics import YOLO` → `YOLO('yolov8n.pt')` — detects cattle (COCO class 19), returns bounding boxes, confidence scores |
| `yolov8n.pt` | Pre-trained YOLOv8 nano model file (6.5 MB) |

---

## Model Architecture Library

### timm (PyTorch Image Models) — v1.0.25
> EfficientNet-B0 architecture with custom classification head.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | `timm.create_model('efficientnet_b0', pretrained=False, num_classes=3)` — loads architecture for inference |
| `scripts/train_cow_classifier_v2.py` | `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3)` — ImageNet pretrained + custom 3-class head |
| `scripts/train_buffalo_classifier.py` | Same — creates EfficientNet-B0 with custom head for 3 buffalo breeds |

---

## Computer Vision

### OpenCV (`cv2`) — v4.11.0
> Image loading, manipulation, and visualization.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | `cv2.imread()` — image loading; `cv2.resize()` — ROI resizing; `cv2.rectangle()` / `cv2.putText()` — drawing detection boxes and labels |
| `app.py` | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` — color space conversion for Streamlit display |

### Pillow (PIL) — v12.0
> High-level image operations.

| Used In | How |
|---------|-----|
| `app.py` | `Image.open()` — loading uploaded images; `image.convert('RGB')` — handling RGBA/grayscale; `image.save()` — saving temp files |
| `scripts/dataset.py` | Image loading for the training dataset pipeline |
| `scripts/multi_view_analysis.py` | Image cropping for three-view analysis (left/front/right) |

---

## Machine Learning Utilities

### scikit-learn — v1.8.0
> Model evaluation metrics.

| Used In | How |
|---------|-----|
| `scripts/evaluate_v2.py` | `sklearn.metrics.classification_report()`, `confusion_matrix()`, `accuracy_score()` — generates per-class precision, recall, F1 |
| `scripts/evaluate_buffalo_model.py` | Same — evaluation metrics for buffalo classifier |

### albumentations — v1.3+
> Advanced data augmentation during training.

| Used In | How |
|---------|-----|
| `scripts/train_cow_classifier_v2.py` | `RandomResizedCrop`, `HorizontalFlip`, `Rotate(15°)`, `ColorJitter`, `Normalize` — augmentation pipeline |
| `scripts/train_buffalo_classifier.py` | Same augmentation strategy for buffalo training |

---

## Data Science & Visualization

### NumPy — v1.26.4
> Numerical array operations.

| Used In | How |
|---------|-----|
| `scripts/inference.py` | Array operations on image data during preprocessing |
| `scripts/generate_charts.py` | `np.array()` for confusion matrices, `np.arange()` for bar chart positioning |
| `scripts/evaluate_v2.py` | Array operations for metric calculations |

### Matplotlib — v3.x
> Static chart and plot generation.

| Used In | How |
|---------|-----|
| `scripts/generate_charts.py` | `plt.subplots()`, `ax.bar()`, `ax.plot()`, `ax.table()`, `ax.annotate()` — generates all 8 comparison charts |
| `scripts/evaluate_v2.py` | Confusion matrix plot saving |
| `scripts/evaluate_buffalo_model.py` | Confusion matrix plot saving |

### Seaborn — v0.13+
> Statistical data visualization.

| Used In | How |
|---------|-----|
| `scripts/generate_charts.py` | `sns.heatmap()` — confusion matrices (chart 5), precision/recall/F1 heatmaps (chart 4) |

### Pandas — v2.x
> Data analysis and manipulation.

| Used In | How |
|---------|-----|
| `scripts/train_cow_classifier_v2.py` | Available for dataset analysis and results tabulation |

---

## Web Application

### Streamlit — v1.55.0
> Interactive web UI for the breed recognition system.

| Used In | How |
|---------|-----|
| `app.py` | `st.set_page_config()` — page title/icon; `st.sidebar.radio()` — model & animal selection; `st.file_uploader()` — image upload; `st.image()` — image display; `st.progress()` — confidence bars; `st.cache_resource` — model caching; `st.columns()` — side-by-side layout |

---

## Progress & Logging

### tqdm — v4.66+
> Progress bars for training loops.

| Used In | How |
|---------|-----|
| `scripts/train_cow_classifier_v2.py` | `tqdm(dataloader)` — progress bar showing batch processing during training epochs |
| `scripts/train_buffalo_classifier.py` | Same — training progress visualization |

### TensorBoard — v2.15+
> Training metrics visualization dashboard.

| Used In | How |
|---------|-----|
| `scripts/train_cow_classifier_v2.py` | Available for logging training/validation loss and accuracy curves |

---

## Infrastructure & DevOps

| Tool | Version | Purpose |
|------|---------|---------|
| **CUDA Toolkit** | 12.4 | GPU compute for PyTorch — `torch.cuda.is_available()` in `inference.py`, `app.py` |
| **NVIDIA Driver** | 582.16 | GPU hardware interface for RTX 4000 Ada |
| **Git** | 2.x | Version control |
| **Git LFS** | 3.7.1 | Large file storage for `.pth`, `.pt`, `.jpg`, `.jpeg`, `.png` on GitHub |
| **venv** | Built-in | Python virtual environment isolation (`setup.bat`) |
| **pip** | 24.0+ | Package installation (`requirements.txt`, `setup.bat`) |

---

## Custom Code (Original Work)

| File | What It Contains |
|------|-----------------|
| `models/custom_efficientnet.py` | **From-scratch EfficientNet** — MBConv blocks with depthwise separable convolutions, Squeeze-and-Excitation (SE) attention, Swish activation, compound scaling, custom classification head |
| `models/custom_resnet.py` | **Custom ResNet32** — BasicBlock with residual connections, batch normalization, custom architecture for cattle classification |
| `scripts/inference.py` | **CattleBreedPredictor** — two-stage pipeline orchestrator: YOLO detection → ROI extraction → EfficientNet/ResNet classification → top-k predictions with confidence scores |
| `scripts/multi_view_analysis.py` | **ThreeViewAnalyzer** — divides detected ROI into left/front/right views for multi-angle breed analysis |
| `scripts/dataset.py` | **CattleBreedDataset** — PyTorch Dataset class with breed-to-label mapping, train/val/test split handling, optional three-view loading |
| `scripts/generate_charts.py` | Chart generator producing 8 comparison visualizations from evaluation metrics |
| `scripts/train_cow_classifier_v2.py` | Full training pipeline — data loading, augmentation, EfficientNet-B0 fine-tuning, class-weighted loss, AdamW + LR scheduling, early stopping, best model saving |
| `scripts/train_buffalo_classifier.py` | Same training architecture adapted for buffalo breeds |
| `app.py` | Streamlit web application integrating all components into an interactive breed recognition tool |

---

## Summary

| Category | Count |
|----------|-------|
| Languages | 4 (Python, JSON, Markdown, Batch) |
| ML/DL Packages | 8 (PyTorch, torchvision, timm, ultralytics, scikit-learn, albumentations, tqdm, TensorBoard) |
| CV Packages | 2 (OpenCV, Pillow) |
| Visualization | 3 (Matplotlib, Seaborn, NumPy) |
| Web Framework | 1 (Streamlit) |
| Data Analysis | 1 (Pandas) |
| Infrastructure | 6 (CUDA, NVIDIA Driver, Git, Git LFS, venv, pip) |
| Custom Scripts | 9 original files |
| **Total Packages** | **15** |
