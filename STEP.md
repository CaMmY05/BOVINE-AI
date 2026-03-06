# 📋 STEP-BY-STEP SETUP GUIDE



---

## ✅ BEFORE YOU START — Check Your System

You need a Windows PC with an **NVIDIA GPU** (any GPU with 4GB+ VRAM works).

**Don't have an NVIDIA GPU?** The project will still work on CPU — just slower inference. Skip CUDA steps.

---

## STEP 1: Install Python 3.11

> ⚠️ Python 3.12+ may cause issues. Use **3.11.x** specifically.

1. Open your browser
2. Go to **https://www.python.org/downloads/release/python-3119/**
3. Scroll down to **Files** section
4. Click **"Windows installer (64-bit)"** to download
5. Run the downloaded `.exe` file
6. **IMPORTANT:** On the first screen, check the box that says **"Add python.exe to PATH"**
7. Click **"Install Now"**
8. Wait for installation to complete
9. Click **"Close"**

### Verify Python is installed:
1. Press `Win + R`, type `cmd`, press Enter
2. Type this exact command and press Enter:
```
py -3.11 --version
```
3. You should see: `Python 3.11.9`

**If you see an error:** Close and reopen cmd. If it still fails, restart your computer and try again.

---

## STEP 2: Install CUDA Toolkit (NVIDIA GPU only)

> Skip this step if you don't have an NVIDIA GPU.

1. Open your browser
2. Go to **https://developer.nvidia.com/cuda-12-4-0-download-archive**
3. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **10** or **11** (your Windows version)
   - Installer Type: **exe (network)** (smaller download)
4. Click **Download**
5. Run the downloaded installer
6. Choose **Express Installation**
7. Wait for it to finish (may take 5-10 minutes)
8. Click **Close**

### Verify CUDA is installed:
1. Open a **new** cmd window (important — must be new!)
2. Type:
```
nvidia-smi
```
3. You should see a table showing your GPU name and "CUDA Version: 12.x"

---

## STEP 3: Install Visual C++ Build Tools

1. Go to **https://visualstudio.microsoft.com/visual-cpp-build-tools/**
2. Click **"Download Build Tools"**
3. Run the downloaded installer
4. In the installer window, check **"Desktop development with C++"**
5. Click **Install** (bottom right)
6. Wait for installation (may take 10-15 minutes, ~6GB download)
7. **Restart your computer** after installation

---

## STEP 4: Install Git

1. Go to **https://git-scm.com/download/win**
2. The download should start automatically (64-bit)
3. Run the installer
4. Click **Next** through all screens (default settings are fine)
5. Click **Install**

### Verify Git is installed:
```
git --version
```
You should see: `git version 2.x.x`

---

## STEP 5: Clone the Repository

1. Open **cmd** (or PowerShell)
2. Navigate to where you want the project (e.g., Desktop):
```
cd %USERPROFILE%\Desktop
```
3. Clone the repo:
```
git clone https://github.com/CaMmY05/BOVINE-AI.git
```
4. Enter the project folder:
```
cd BOVINE-AI
```

> **Note:** Replace `YOUR_USERNAME` with the actual GitHub username.

---

## STEP 6: Create Virtual Environment

1. Make sure you're inside the project folder (`BOVINE-AI` or `FINAL-MVP-CATTLE`)
2. Run:
```
py -3.11 -m venv venv
```
3. Wait 10-20 seconds for it to create

### Verify the venv was created:
You should see a new `venv` folder inside the project directory.

---

## STEP 7: Activate the Virtual Environment

```
venv\Scripts\activate
```

After running this, your command prompt should show `(venv)` at the beginning:
```
(venv) C:\Users\YourName\Desktop\BOVINE-AI>
```

> **IMPORTANT:** Every time you open a new terminal to work with this project, you must run `venv\Scripts\activate` first!

---

## STEP 8: Install PyTorch with CUDA Support

**If you have an NVIDIA GPU:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**If you do NOT have an NVIDIA GPU (CPU only):**
```
pip install torch torchvision torchaudio
```

> This download is ~2.5 GB. Wait for it to complete. It may take 5-15 minutes depending on your internet speed.

### Verify PyTorch:
```
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output (with GPU):
```
PyTorch: 2.6.0+cu124
CUDA: True
```

Expected output (CPU only):
```
PyTorch: 2.6.0
CUDA: False
```

---

## STEP 9: Install Remaining Dependencies

```
pip install -r requirements.txt
```

Wait for all packages to download and install (2-5 minutes).

### Verify everything is installed:
```
python -c "import timm; import ultralytics; import streamlit; import cv2; print('All packages OK!')"
```

You should see: `All packages OK!`

---

## STEP 10: Run the Web Application

```
streamlit run app.py
```

**First time only:** Streamlit will ask for your email. Just press **Enter** to skip.

You should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## STEP 11: Open the App in Your Browser

1. Open any browser (Chrome, Edge, Firefox)
2. Go to: **http://localhost:8501**
3. You should see the **"🐄 Cattle Breed Recognition System"** page

---

## STEP 12: Test the Application

### Test Cow Detection:
1. In the sidebar, select **"EfficientNet-B0 V2"** (should be default)
2. Select **"🐄 Cow"**
3. Click **"Browse files"** under "Choose an image..."
4. Navigate to: `data\processed_v2\cows\test\gir\`
5. Select any `.jpg` image
6. Wait 2-3 seconds
7. You should see:
   - ✅ The original image on the left
   - ✅ Detection results with bounding box on the right
   - ✅ Breed prediction showing "Gir" with ~99% confidence

### Test Buffalo Detection:
1. In the sidebar, switch to **"🐃 Buffalo"**
2. Upload an image from: `data\processed_v2\buffaloes\test\murrah\`
3. You should see "Murrah" predicted with high confidence

### Test Different Models:
1. Try switching to **"ResNet18"** in the sidebar
2. Upload another image
3. Compare the confidence scores with EfficientNet-B0 V2

---

## STEP 13: View the Comparison Charts

All charts are pre-generated in the `charts/` folder:

```
charts/
├── 01_overall_accuracy_comparison.png    ← Model accuracy comparison
├── 02_per_breed_cow_accuracy.png         ← Per-breed cow results
├── 03_f1_score_comparison.png            ← F1 score comparison
├── 04_precision_recall_f1_heatmap.png    ← Precision/Recall heatmaps
├── 05_confusion_matrices.png            ← Confusion matrices
├── 06_training_curves.png               ← Training loss/accuracy curves
├── 07_model_architecture_comparison.png ← Architecture comparison table
├── 08_improvement_journey.png           ← Accuracy improvement journey
```

Double-click any `.png` file to open it.

**To regenerate charts** (optional):
```
python -X utf8 scripts/generate_charts.py
```

---

## STEP 14: Explore the Custom Model Code

### Custom EfficientNet (from-scratch implementation):
Open `models/custom_efficientnet.py` to see:
- MBConv blocks with depthwise separable convolutions
- Squeeze-and-Excitation attention mechanism
- Custom classification head

### Custom ResNet:
Open `models/custom_resnet.py` to see:
- ResNet32 implementation with BasicBlock
- Custom architecture for cattle classification

### Inference Pipeline:
Open `scripts/inference.py` to see:
- `CattleBreedPredictor` class
- Two-stage detection → classification pipeline
- YOLO detection + EfficientNet classification integration

---

## STEP 15: Stop the Application

To stop Streamlit, go back to the terminal and press:
```
Ctrl + C
```

---

## 🔄 NEXT TIME: How to Start Again

Every time you want to run the project after closing the terminal:

```bash
# 1. Open cmd/PowerShell
# 2. Navigate to project folder
cd C:\path\to\BOVINE-AI

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Run the app
streamlit run app.py

# 5. Open http://localhost:8501 in browser
```

---

## ❌ TROUBLESHOOTING

### "Python not found" or "py is not recognized"
- Reinstall Python 3.11 and make sure to check **"Add to PATH"**
- Restart your computer after installing

### "No module named torch" or other import errors
- Make sure you activated the virtual environment: `venv\Scripts\activate`
- You should see `(venv)` at the start of your command prompt

### "CUDA not available" (but you have an NVIDIA GPU)
- Make sure you installed CUDA Toolkit 12.4 (Step 2)
- Make sure you installed PyTorch with `--index-url https://download.pytorch.org/whl/cu124`
- Try: `nvidia-smi` — if this fails, update your NVIDIA drivers

### "No cattle detected in the image"
- Try lowering the **Detection Confidence** slider in the sidebar (set to 0.1 or 0.2)
- Make sure the image contains a clearly visible cow or buffalo
- Use test images from `data/processed_v2/cows/test/` for guaranteed results

### "Model not found" error
- Make sure you're running from the project root folder (where `app.py` is)
- Check that `models/classification/cow_classifier_v2/best_model.pth` exists

### Streamlit shows a blank page
- Wait 5-10 seconds for it to load
- Try refreshing the page (F5)
- Check the terminal for error messages

### "pip install" fails with build errors
- Make sure Visual C++ Build Tools is installed (Step 3)
- Try upgrading pip: `python -m pip install --upgrade pip`

---


