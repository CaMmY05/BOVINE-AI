"""
Download ALL datasets identified in Parallel.ai research
This will download datasets for future use
"""

import subprocess
import os
from pathlib import Path
import time

def download_kaggle_dataset(dataset_name, output_dir, description):
    """Download a Kaggle dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        print("Downloading... (this may take a few minutes)")
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', output_dir, '--unzip'],
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout
        )
        
        print(f"✓ Downloaded successfully!")
        
        # Count files
        file_count = sum(1 for _ in Path(output_dir).rglob('*') if _.is_file())
        print(f"✓ Total files: {file_count}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: Download took too long")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading: {e}")
        if e.stdout:
            print(f"  Output: {e.stdout}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def download_roboflow_dataset(project_url, workspace, project, output_dir, description):
    """Instructions for downloading Roboflow dataset"""
    print(f"\n{'='*60}")
    print(f"Roboflow Dataset: {description}")
    print(f"{'='*60}")
    print(f"URL: {project_url}")
    print()
    print("To download this dataset:")
    print("1. Visit the URL above")
    print("2. Click 'Download Dataset'")
    print("3. Select format: 'Folder Structure' or 'YOLO'")
    print("4. Download to:", output_dir)
    print()
    print("Or use Roboflow CLI:")
    print(f"   roboflow download -w {workspace} -p {project}")
    print()
    input("Press Enter when downloaded (or skip)...")

def main():
    """Download all research datasets"""
    print("="*60)
    print("DOWNLOADING ALL RESEARCH DATASETS")
    print("From Parallel.ai Research Report")
    print("="*60)
    
    # Kaggle datasets to download
    kaggle_datasets = [
        {
            'name': 'sujayroy723/indian-cattle-breeds',
            'description': 'Indian Cattle Breeds (5,949 images, balanced)',
            'output': 'data/research_datasets/kaggle/indian_cattle_breeds',
            'priority': 'HIGH',
            'notes': 'Best for balanced baseline - 100 images per breed'
        }
        # Note: lukex9442/indian-bovine-breeds already downloaded
    ]
    
    # Roboflow datasets (manual download)
    roboflow_datasets = [
        {
            'url': 'https://universe.roboflow.com/shiv-q9erb/indian-bovine-breed-recognition-hen07',
            'workspace': 'shiv-q9erb',
            'project': 'indian-bovine-breed-recognition-hen07',
            'description': 'Indian Bovine Breed Recognition (5,723 images)',
            'output': 'data/research_datasets/roboflow/indian_bovine_recognition',
            'priority': 'HIGH',
            'notes': 'Most comprehensive, excellent for Gir augmentation'
        },
        {
            'url': 'https://universe.roboflow.com/breeddetection/cattle-breed-9rfl6',
            'workspace': 'breeddetection',
            'project': 'cattle-breed-9rfl6',
            'description': 'Cattle Breed Object Detection (2,017 images)',
            'output': 'data/research_datasets/roboflow/cattle_breed_detection',
            'priority': 'HIGH',
            'notes': 'Essential for two-stage classifier'
        },
        {
            'url': 'https://universe.roboflow.com/object-detection-zrnsd/red_sindhi-ybeen',
            'workspace': 'object-detection-zrnsd',
            'project': 'red_sindhi-ybeen',
            'description': 'Red_Sindhi Object Detection (165 images)',
            'output': 'data/research_datasets/roboflow/red_sindhi',
            'priority': 'CRITICAL',
            'notes': 'Focused Red Sindhi dataset - PRIORITY!'
        },
        {
            'url': 'https://universe.roboflow.com/final-bwjlq/sahiwal-cow-onsxx',
            'workspace': 'final-bwjlq',
            'project': 'sahiwal-cow-onsxx',
            'description': 'Sahiwal Cow Object Detection (104 images)',
            'output': 'data/research_datasets/roboflow/sahiwal',
            'priority': 'MEDIUM',
            'notes': 'Focused Sahiwal dataset'
        },
        {
            'url': 'https://universe.roboflow.com/cowbreed/cow-breeds-zwbex',
            'workspace': 'cowbreed',
            'project': 'cow-breeds-zwbex',
            'description': 'Cow Breeds Object Detection (98 images)',
            'output': 'data/research_datasets/roboflow/cow_breeds',
            'priority': 'LOW',
            'notes': 'Small but includes all 3 breeds'
        },
        {
            'url': 'https://universe.roboflow.com/annotations-kyert/kaggle-breed',
            'workspace': 'annotations-kyert',
            'project': 'kaggle-breed',
            'description': 'kaggle-breed Classification (5,825 images)',
            'output': 'data/research_datasets/roboflow/kaggle_breed',
            'priority': 'MEDIUM',
            'notes': 'Large classification dataset'
        }
    ]
    
    # Academic datasets (manual contact required)
    academic_datasets = [
        {
            'name': 'Cowbree Dataset',
            'images': '4,000 (1,193 for Sahiwal + Red Sindhi)',
            'source': 'Academic Paper',
            'url': 'https://beei.org/index.php/EEI/article/download/2443/1802',
            'priority': 'GOLD STANDARD',
            'notes': 'Expert-validated, highest quality. Contact authors for access.'
        },
        {
            'name': 'Indigenous Cattle ID (KrishiKosh)',
            'images': '480 high-res (240 Sahiwal, 240 Red Sindhi)',
            'source': 'KrishiKosh Thesis',
            'url': 'https://krishikosh.egranth.ac.in/items/4ca5ec28-a558-406a-aca6-64449d724422',
            'priority': 'GOLD STANDARD',
            'notes': 'High-resolution (4080x2296), controlled capture. Contact authors.'
        }
    ]
    
    print("\n" + "="*60)
    print("DATASETS TO DOWNLOAD")
    print("="*60)
    
    print("\n📦 KAGGLE DATASETS (Automated):")
    for idx, ds in enumerate(kaggle_datasets, 1):
        print(f"\n{idx}. {ds['description']}")
        print(f"   Priority: {ds['priority']}")
        print(f"   Notes: {ds['notes']}")
    
    print("\n🌐 ROBOFLOW DATASETS (Manual/Semi-automated):")
    for idx, ds in enumerate(roboflow_datasets, 1):
        print(f"\n{idx}. {ds['description']}")
        print(f"   Priority: {ds['priority']}")
        print(f"   Notes: {ds['notes']}")
    
    print("\n🎓 ACADEMIC DATASETS (Contact Required):")
    for idx, ds in enumerate(academic_datasets, 1):
        print(f"\n{idx}. {ds['name']}")
        print(f"   Images: {ds['images']}")
        print(f"   Priority: {ds['priority']}")
        print(f"   Notes: {ds['notes']}")
    
    print("\n" + "="*60)
    print("DOWNLOAD OPTIONS")
    print("="*60)
    print("1. Download Kaggle datasets only (automated)")
    print("2. Show Roboflow download instructions")
    print("3. Show Academic dataset contact info")
    print("4. Download ALL (Kaggle + instructions for others)")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Download Kaggle datasets
        print("\n🚀 Downloading Kaggle datasets...")
        success_count = 0
        for ds in kaggle_datasets:
            if download_kaggle_dataset(ds['name'], ds['output'], ds['description']):
                success_count += 1
            time.sleep(2)  # Be nice to Kaggle servers
        
        print(f"\n✓ Downloaded {success_count}/{len(kaggle_datasets)} Kaggle datasets")
    
    elif choice == '2':
        # Show Roboflow instructions
        print("\n🌐 ROBOFLOW DOWNLOAD INSTRUCTIONS")
        for ds in roboflow_datasets:
            download_roboflow_dataset(
                ds['url'], ds['workspace'], ds['project'], 
                ds['output'], ds['description']
            )
    
    elif choice == '3':
        # Show academic dataset info
        print("\n🎓 ACADEMIC DATASETS - CONTACT INFORMATION")
        print("="*60)
        
        print("\n1. COWBREE DATASET")
        print("   Paper: https://beei.org/index.php/EEI/article/download/2443/1802")
        print("   Contact: Check paper for author emails")
        print("   Email template: See ACADEMIC_DATASET_GUIDE.md")
        
        print("\n2. KRISHIKOSH THESIS")
        print("   URL: https://krishikosh.egranth.ac.in/items/4ca5ec28-a558-406a-aca6-64449d724422")
        print("   Contact: Check thesis for author contact")
        print("   Email template: See ACADEMIC_DATASET_GUIDE.md")
        
        print("\nI'll create a detailed guide for contacting authors...")
    
    elif choice == '4':
        # Download all
        print("\n🚀 DOWNLOADING ALL DATASETS")
        
        # Kaggle
        print("\n" + "="*60)
        print("PART 1: KAGGLE DATASETS")
        print("="*60)
        success_count = 0
        for ds in kaggle_datasets:
            if download_kaggle_dataset(ds['name'], ds['output'], ds['description']):
                success_count += 1
            time.sleep(2)
        
        print(f"\n✓ Downloaded {success_count}/{len(kaggle_datasets)} Kaggle datasets")
        
        # Roboflow
        print("\n" + "="*60)
        print("PART 2: ROBOFLOW DATASETS")
        print("="*60)
        print("\nRoboflow datasets require manual download.")
        show_roboflow = input("Show instructions? (y/n): ").strip().lower()
        if show_roboflow == 'y':
            for ds in roboflow_datasets:
                download_roboflow_dataset(
                    ds['url'], ds['workspace'], ds['project'],
                    ds['output'], ds['description']
                )
        
        # Academic
        print("\n" + "="*60)
        print("PART 3: ACADEMIC DATASETS")
        print("="*60)
        print("\nAcademic datasets require contacting authors.")
        print("I'll create a detailed guide for you...")
    
    else:
        print("Exiting...")
        return
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print("\nDatasets downloaded/ready:")
    print("✓ Original dataset: 947 images (restored)")
    print("✓ Kaggle datasets: Check data/research_datasets/kaggle/")
    print("⏳ Roboflow datasets: Manual download required")
    print("⏳ Academic datasets: Contact authors")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review downloaded datasets")
    print("2. Clean and organize images")
    print("3. Selectively add high-quality images")
    print("4. Retrain model")

if __name__ == "__main__":
    main()
