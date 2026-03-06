"""
Sample script to download cattle images from various sources
This is a helper script to get started quickly with the MVP
"""

import os
import requests
from pathlib import Path
import json

def download_from_url(url, save_path):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_sample_images():
    """
    Download sample cattle images for testing
    Note: These are just examples. For production, use proper datasets.
    """
    
    print("Downloading sample cattle images...")
    print("Note: For a real MVP, please use proper cattle breed datasets from:")
    print("  - Roboflow: https://universe.roboflow.com/")
    print("  - Kaggle: https://www.kaggle.com/")
    print("  - Bristol Dataset: https://data.bris.ac.uk/")
    
    # Sample image URLs (replace with actual cattle images)
    # These are placeholder URLs - you need to replace with actual cattle images
    sample_images = {
        'gir': [
            # Add actual Gir cattle image URLs here
        ],
        'sahiwal': [
            # Add actual Sahiwal cattle image URLs here
        ],
        'red_sindhi': [
            # Add actual Red Sindhi cattle image URLs here
        ],
        'murrah_buffalo': [
            # Add actual Murrah buffalo image URLs here
        ],
        'mehsana_buffalo': [
            # Add actual Mehsana buffalo image URLs here
        ]
    }
    
    base_dir = Path('data/raw')
    
    for breed, urls in sample_images.items():
        breed_dir = base_dir / breed
        breed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {breed} images...")
        for idx, url in enumerate(urls, 1):
            save_path = breed_dir / f"{breed}_{idx:03d}.jpg"
            if download_from_url(url, str(save_path)):
                print(f"  ✓ Downloaded {save_path.name}")
            else:
                print(f"  ✗ Failed to download image {idx}")
    
    print("\n" + "="*60)
    print("IMPORTANT: Sample images are for demonstration only!")
    print("="*60)
    print("\nFor a real MVP, please:")
    print("1. Collect your own cattle images")
    print("2. Or download from proper datasets (Roboflow, Kaggle)")
    print("3. Organize them in data/raw/<breed_name>/ folders")
    print("4. Ensure you have at least 50-100 images per breed")

def setup_kaggle_download():
    """
    Instructions for downloading from Kaggle
    """
    print("\n" + "="*60)
    print("HOW TO DOWNLOAD FROM KAGGLE")
    print("="*60)
    print("\n1. Install Kaggle CLI:")
    print("   pip install kaggle")
    
    print("\n2. Get your API token:")
    print("   - Go to https://www.kaggle.com/")
    print("   - Click on your profile picture → Account")
    print("   - Scroll to 'API' section")
    print("   - Click 'Create New API Token'")
    print("   - This downloads kaggle.json")
    
    print("\n3. Place kaggle.json in:")
    print("   Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
    print("   Linux/Mac: ~/.kaggle/kaggle.json")
    
    print("\n4. Download cattle dataset:")
    print("   kaggle datasets download -d vikramamin/cattle-breed-classification-dataset")
    print("   unzip cattle-breed-classification-dataset.zip -d data/raw/")
    
    print("\n5. Search for more datasets:")
    print("   kaggle datasets list -s 'cattle breed'")

def setup_roboflow_download():
    """
    Instructions for downloading from Roboflow
    """
    print("\n" + "="*60)
    print("HOW TO DOWNLOAD FROM ROBOFLOW")
    print("="*60)
    print("\n1. Visit: https://universe.roboflow.com/")
    
    print("\n2. Search for 'cattle detection' or 'cow detection'")
    
    print("\n3. Select a dataset (look for ones with good ratings)")
    
    print("\n4. Click 'Download Dataset'")
    
    print("\n5. Choose format: 'Folder Structure' or 'YOLO'")
    
    print("\n6. Download and extract to data/raw/")
    
    print("\n7. Reorganize if needed to match structure:")
    print("   data/raw/")
    print("   ├── breed1/")
    print("   │   └── *.jpg")
    print("   ├── breed2/")
    print("   │   └── *.jpg")
    print("   └── ...")

def create_dummy_dataset():
    """
    Create a minimal dummy dataset for testing the pipeline
    """
    print("\n" + "="*60)
    print("CREATING DUMMY DATASET FOR TESTING")
    print("="*60)
    
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    breeds = ['gir', 'sahiwal', 'red_sindhi', 'murrah_buffalo', 'mehsana_buffalo']
    base_dir = Path('data/raw')
    
    for breed in breeds:
        breed_dir = base_dir / breed
        breed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 dummy images per breed
        for i in range(10):
            # Create a colored image
            img = Image.new('RGB', (640, 480), 
                          color=(random.randint(100, 200), 
                                random.randint(100, 200), 
                                random.randint(100, 200)))
            
            draw = ImageDraw.Draw(img)
            
            # Draw a simple cattle-like shape (rectangle + circle for head)
            # Body
            draw.rectangle([200, 200, 440, 350], fill='brown', outline='black', width=3)
            # Head
            draw.ellipse([420, 220, 500, 300], fill='brown', outline='black', width=3)
            # Legs
            draw.rectangle([220, 350, 250, 450], fill='brown', outline='black', width=2)
            draw.rectangle([290, 350, 320, 450], fill='brown', outline='black', width=2)
            draw.rectangle([360, 350, 390, 450], fill='brown', outline='black', width=2)
            draw.rectangle([410, 350, 440, 450], fill='brown', outline='black', width=2)
            
            # Add breed label
            draw.text((250, 100), f"{breed.upper()}", fill='white')
            draw.text((250, 130), f"Sample {i+1}", fill='white')
            
            # Save
            save_path = breed_dir / f"{breed}_{i+1:03d}.jpg"
            img.save(save_path)
        
        print(f"✓ Created 10 dummy images for {breed}")
    
    print("\n" + "="*60)
    print("DUMMY DATASET CREATED!")
    print("="*60)
    print("\nThis is just for testing the pipeline.")
    print("Replace with real cattle images for actual training!")
    print("\nNext steps:")
    print("1. python scripts/prepare_data.py")
    print("2. python scripts/train_classifier.py")

if __name__ == "__main__":
    print("="*60)
    print("CATTLE BREED MVP - DATA DOWNLOAD HELPER")
    print("="*60)
    
    print("\nChoose an option:")
    print("1. Create dummy dataset (for testing pipeline)")
    print("2. Show Kaggle download instructions")
    print("3. Show Roboflow download instructions")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        create_dummy_dataset()
    elif choice == '2':
        setup_kaggle_download()
    elif choice == '3':
        setup_roboflow_download()
    else:
        print("Exiting...")
    
    print("\n" + "="*60)
    print("For more information, see README.md and QUICKSTART.md")
    print("="*60)
