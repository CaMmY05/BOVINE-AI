import streamlit as st
import torch
from PIL import Image
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Page config
st.set_page_config(
    page_title="Cattle Breed Recognition - FINAL MVP",
    page_icon="🐄",
    layout="wide"
)

# Title and description
st.title("🐄 Cattle Breed Recognition System")
st.markdown("""
**FINAL MVP** — YOLO detection + Deep Learning breed classification.
Upload an image of **cattle (cows)** or **buffaloes** to detect and classify the breed.

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Cow Classifier V2 | EfficientNet-B0 | **98.85%** |
| Buffalo Classifier V1 | EfficientNet-B0 | **95.96%** |
""")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Model architecture selector
model_arch = st.sidebar.radio(
    "Select Model Architecture:",
    ["EfficientNet-B0 V2", "ResNet18", "ResNet34"],
    index=0
)

# Map model architecture to supported animal types
MODEL_ANIMAL_SUPPORT = {
    "EfficientNet-B0 V2": ["🐄 Cow", "🐃 Buffalo"],
    "ResNet18": ["🐄 Cow", "🐃 Buffalo"],
    "ResNet34": ["🐄 Cow", "🐃 Buffalo"]
}

# Get supported animal types for selected model
supported_animals = MODEL_ANIMAL_SUPPORT[model_arch]

# If current selection is not in supported animals, default to first supported
current_animal = st.session_state.get('animal_type', supported_animals[0])
if current_animal not in supported_animals:
    current_animal = supported_animals[0]

# Animal type selector
animal_type = st.sidebar.radio(
    "Select Animal Type:",
    supported_animals,
    index=supported_animals.index(current_animal) if current_animal in supported_animals else 0
)

# Store current selection in session state
st.session_state['animal_type'] = animal_type

# Map model architecture to directory names
MODEL_PATHS = {
    "EfficientNet-B0 V2": {
        "Cow": "cow_classifier_v2",
        "Buffalo": "buffalo_classifier_v1"
    },
    "ResNet18": {
        "Cow": "resnet18_cow_v1",
        "Buffalo": "resnet18_buffalo_v1"
    },
    "ResNet34": {
        "Cow": "resnet34_cow_v1",
        "Buffalo": "resnet34_buffalo_v1"
    }
}

# Map architecture names to model types
ARCH_TO_MODEL = {
    "EfficientNet-B0 V2": "efficientnet_b0",
    "ResNet18": "resnet18",
    "ResNet34": "resnet32"
}

# Get base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the model directory
animal_key = animal_type.replace("🐄 ", "").replace("🐃 ", "")
model_dir = MODEL_PATHS[model_arch][animal_key]
model_base_path = os.path.join(BASE_DIR, "models", "classification", model_dir)
classes_path = os.path.join(model_base_path, "classes.json")

# For ResNet models, we need to look in checkpoints directory
if model_arch in ["ResNet18", "ResNet34"]:
    if os.path.exists(os.path.join(model_base_path, "checkpoints", "best_model.pth")):
        model_path = os.path.join(model_base_path, "checkpoints", "best_model.pth")
    elif os.path.exists(os.path.join(model_base_path, "best_model.pth")):
        model_path = os.path.join(model_base_path, "best_model.pth")
    else:
        model_path = os.path.join(model_base_path, "final_model.pth")
else:
    if os.path.exists(os.path.join(model_base_path, "best_model.pth")):
        model_path = os.path.join(model_base_path, "best_model.pth")
    else:
        model_path = os.path.join(model_base_path, "final_model.pth")

# Set model version for display
model_version = f"{model_arch} ({animal_key})"

# Check if models exist
detection_model_path = os.path.join(BASE_DIR, 'yolov8n.pt')
models_exist = os.path.exists(model_path) and os.path.exists(classes_path)

if not models_exist:
    st.warning(f"⚠️ Model not found at `{model_path}`.")
    st.info("""
    **Available Models:**
    - EfficientNet-B0 V2: cow_classifier_v2/ & buffalo_classifier_v1/
    - ResNet18: resnet18_cow_v1/ & resnet18_buffalo_v1/
    - ResNet34: resnet34_cow_v1/ & resnet34_buffalo_v1/
    """)
    st.stop()

# Set classification model path
classification_model_path = model_path

# Load model (cached with animal type and architecture as key)
@st.cache_resource
def load_predictor(cache_key, det_path, cls_path, cls_json_path, arch_name):
    from scripts.inference import CattleBreedPredictor
    model_arch_str = ARCH_TO_MODEL[arch_name]
    return CattleBreedPredictor(
        detection_model_path=det_path,
        classification_model_path=cls_path,
        classes_path=cls_json_path,
        model_arch=model_arch_str
    )

try:
    predictor = load_predictor(
        f"{animal_type}_{model_arch}", 
        detection_model_path,
        classification_model_path, 
        classes_path,
        model_arch
    )
    st.sidebar.success(f"✓ Models loaded successfully!")
    st.sidebar.info(f"**Active Model:** {model_version}")
    
    # Device info
    device = "GPU 🚀" if torch.cuda.is_available() else "CPU"
    st.sidebar.caption(f"Running on: {device}")
    
    # Settings
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    show_three_views = st.sidebar.checkbox("Show Three-View Analysis", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Save temporarily
        temp_path = os.path.join(BASE_DIR, "temp_upload.jpg")
        image.save(temp_path, 'JPEG')
        
        # Run prediction
        with st.spinner("Detecting and classifying..."):
            try:
                rois, boxes_info = predictor.detect_and_extract_roi(temp_path, confidence_threshold)
                
                if not rois:
                    st.error("❌ No cattle detected in the image. Try adjusting the confidence threshold.")
                else:
                    st.success(f"✓ Detected {len(rois)} cattle")
                    
                    for idx, (roi, box_info) in enumerate(zip(rois, boxes_info)):
                        st.markdown(f"### 🐄 Cattle #{idx+1}")
                        
                        predictions = predictor.classify_breed(roi, top_k=3)
                        
                        roi_col, pred_col = st.columns([1, 2])
                        
                        with roi_col:
                            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), 
                                   caption=f"Detected ROI #{idx+1}", 
                                   use_container_width=True)
                        
                        with pred_col:
                            st.markdown("**Top Predictions:**")
                            for rank, pred in enumerate(predictions, 1):
                                confidence = pred['score']
                                breed = pred['breed'].replace('_', ' ').title()
                                
                                st.markdown(f"**{rank}. {breed}**")
                                st.progress(confidence / 100)
                                st.caption(f"Confidence: {confidence:.2f}%")
                        
                        # Three-view analysis
                        if show_three_views:
                            st.markdown("**Three-View Analysis:**")
                            from scripts.multi_view_analysis import ThreeViewAnalyzer
                            analyzer = ThreeViewAnalyzer()
                            
                            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            views = analyzer.divide_into_views(roi_pil)
                            
                            view_cols = st.columns(3)
                            for view_idx, (view_name, view_img) in enumerate(views.items()):
                                with view_cols[view_idx]:
                                    st.image(view_img, caption=view_name.capitalize(), use_container_width=True)
                        
                        st.markdown("---")
                    
                    # Visualize on original image
                    with col2:
                        st.subheader("Detection Results")
                        
                        img_cv = cv2.imread(temp_path)
                        for idx, (box_info, roi) in enumerate(zip(boxes_info, rois)):
                            x1, y1, x2, y2 = box_info['bbox']
                            
                            predictions = predictor.classify_breed(roi, top_k=1)
                            top_breed = predictions[0]
                            
                            color = (0, 255, 0)
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                            
                            label = f"{top_breed['breed']}: {top_breed['score']:.1f}%"
                            (label_width, label_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                            )
                            cv2.rectangle(img_cv, (x1, y1 - label_height - 10), 
                                        (x1 + label_width, y1), color, -1)
                            cv2.putText(img_cv, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Additional info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.info("""
    **Cow Breeds (98.85%):**
    - Gir (99.72%)
    - Sahiwal (99.31%)
    - Red Sindhi (95.60%)
    
    **Buffalo Breeds (95.96%):**
    - Jaffarabadi (100.00%)
    - Murrah (97.83%)
    - Mehsana (87.50%)
    """)

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.exception(e)
