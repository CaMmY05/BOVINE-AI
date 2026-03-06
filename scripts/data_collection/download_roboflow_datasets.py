"""
Download Roboflow datasets using Roboflow CLI
Automated download, extraction, and organization
"""

import subprocess
import os
import shutil
from pathlib import Path
import json

def install_roboflow():
    """Install roboflow package if not already installed"""
    print("="*60)
    print("CHECKING ROBOFLOW INSTALLATION")
    print("="*60)
    
    try:
        import roboflow
        print("✓ Roboflow already installed")
        return True
    except ImportError:
        print("Installing roboflow package...")
        try:
            subprocess.run(['pip', 'install', 'roboflow'], check=True)
            print("✓ Roboflow installed successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to install roboflow: {e}")
            return False

def download_roboflow_dataset(workspace, project, version, output_dir, description, api_key=None):
    """Download a Roboflow dataset using Python API"""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"Project: {workspace}/{project}")
    print(f"{'='*60}")
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow
        if api_key:
            rf = Roboflow(api_key=api_key)
        else:
            # Will prompt for API key if not provided
            rf = Roboflow()
        
        # Get project
        project_obj = rf.workspace(workspace).project(project)
        
        # Download dataset
        print(f"Downloading version {version}...")
        dataset = project_obj.version(version).download(
            model_format="folder",  # or "yolov5", "coco", etc.
            location=output_dir
        )
        
        print(f"✓ Downloaded successfully to: {output_dir}")
        
        # Count files
        file_count = sum(1 for _ in Path(output_dir).rglob('*') if _.is_file() and _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
        print(f"✓ Total images: {file_count}")
        
        return True, file_count
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False, 0

def organize_roboflow_data(source_dir, breed_name, dest_base='data/research_datasets/roboflow_organized'):
    """Organize downloaded Roboflow data into breed folders"""
    print(f"\nOrganizing {breed_name} data...")
    
    source_path = Path(source_dir)
    dest_path = Path(dest_base) / breed_name
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images in train/valid/test folders
    image_count = 0
    for split in ['train', 'valid', 'test']:
        split_dir = source_path / split / 'images'
        if split_dir.exists():
            for img_file in split_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copy to organized folder
                    dest_file = dest_path / f"{split}_{img_file.name}"
                    shutil.copy2(img_file, dest_file)
                    image_count += 1
    
    print(f"✓ Organized {image_count} images to: {dest_path}")
    return image_count

def main():
    """Download all priority Roboflow datasets"""
    print("="*60)
    print("ROBOFLOW DATASET DOWNLOADER")
    print("Using Roboflow Python API")
    print("="*60)
    
    # Install roboflow if needed
    if not install_roboflow():
        print("\n✗ Cannot proceed without roboflow package")
        return
    
    # Roboflow datasets to download
    datasets = [
        {
            'workspace': 'object-detection-zrnsd',
            'project': 'red_sindhi-ybeen',
            'version': 1,  # Usually 1, check on Roboflow
            'description': 'Red_Sindhi Object Detection (165 images)',
            'output': 'data/research_datasets/roboflow/red_sindhi',
            'breed': 'red_sindhi',
            'priority': 'CRITICAL'
        },
        {
            'workspace': 'shiv-q9erb',
            'project': 'indian-bovine-breed-recognition-hen07',
            'version': 1,
            'description': 'Indian Bovine Breed Recognition (5,723 images)',
            'output': 'data/research_datasets/roboflow/indian_bovine_recognition',
            'breed': 'multi',
            'priority': 'HIGH'
        },
        {
            'workspace': 'breeddetection',
            'project': 'cattle-breed-9rfl6',
            'version': 1,
            'description': 'Cattle Breed Object Detection (2,017 images)',
            'output': 'data/research_datasets/roboflow/cattle_breed_detection',
            'breed': 'multi',
            'priority': 'HIGH'
        },
        {
            'workspace': 'final-bwjlq',
            'project': 'sahiwal-cow-onsxx',
            'version': 1,
            'description': 'Sahiwal Cow Object Detection (104 images)',
            'output': 'data/research_datasets/roboflow/sahiwal',
            'breed': 'sahiwal',
            'priority': 'MEDIUM'
        },
        {
            'workspace': 'cowbreed',
            'project': 'cow-breeds-zwbex',
            'version': 1,
            'description': 'Cow Breeds Object Detection (98 images)',
            'output': 'data/research_datasets/roboflow/cow_breeds',
            'breed': 'multi',
            'priority': 'LOW'
        },
        {
            'workspace': 'annotations-kyert',
            'project': 'kaggle-breed',
            'version': 1,
            'description': 'kaggle-breed Classification (5,825 images)',
            'output': 'data/research_datasets/roboflow/kaggle_breed',
            'breed': 'multi',
            'priority': 'MEDIUM'
        }
    ]
    
    print("\nDatasets to download:")
    for idx, ds in enumerate(datasets, 1):
        print(f"\n{idx}. {ds['description']}")
        print(f"   Priority: {ds['priority']}")
        print(f"   Breed: {ds['breed']}")
    
    print("\n" + "="*60)
    print("DOWNLOAD OPTIONS")
    print("="*60)
    print("1. Download Priority datasets only (Red_Sindhi + 2 HIGH)")
    print("2. Download ALL datasets")
    print("3. Download specific dataset (choose number)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    # Get API key
    print("\n" + "="*60)
    print("ROBOFLOW API KEY")
    print("="*60)
    print("You need a Roboflow API key to download datasets.")
    print("Get it from: https://app.roboflow.com/settings/api")
    print()
    api_key = input("Enter your Roboflow API key (or press Enter to be prompted): ").strip()
    if not api_key:
        api_key = None
    
    # Download based on choice
    downloaded = []
    
    if choice == '1':
        # Priority datasets
        priority_datasets = [ds for ds in datasets if ds['priority'] in ['CRITICAL', 'HIGH']]
        print(f"\n🚀 Downloading {len(priority_datasets)} priority datasets...")
        
        for ds in priority_datasets:
            success, count = download_roboflow_dataset(
                ds['workspace'], ds['project'], ds['version'],
                ds['output'], ds['description'], api_key
            )
            if success:
                downloaded.append({'name': ds['description'], 'count': count, 'breed': ds['breed'], 'output': ds['output']})
    
    elif choice == '2':
        # All datasets
        print(f"\n🚀 Downloading ALL {len(datasets)} datasets...")
        
        for ds in datasets:
            success, count = download_roboflow_dataset(
                ds['workspace'], ds['project'], ds['version'],
                ds['output'], ds['description'], api_key
            )
            if success:
                downloaded.append({'name': ds['description'], 'count': count, 'breed': ds['breed'], 'output': ds['output']})
    
    elif choice == '3':
        # Specific dataset
        print("\nAvailable datasets:")
        for idx, ds in enumerate(datasets, 1):
            print(f"{idx}. {ds['description']}")
        
        dataset_num = int(input("\nEnter dataset number: ").strip()) - 1
        if 0 <= dataset_num < len(datasets):
            ds = datasets[dataset_num]
            success, count = download_roboflow_dataset(
                ds['workspace'], ds['project'], ds['version'],
                ds['output'], ds['description'], api_key
            )
            if success:
                downloaded.append({'name': ds['description'], 'count': count, 'breed': ds['breed'], 'output': ds['output']})
    
    else:
        print("Exiting...")
        return
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    if downloaded:
        print(f"\n✓ Successfully downloaded {len(downloaded)} datasets:")
        total_images = 0
        for ds in downloaded:
            print(f"\n  {ds['name']}")
            print(f"    Images: {ds['count']}")
            print(f"    Breed: {ds['breed']}")
            print(f"    Location: {ds['output']}")
            total_images += ds['count']
        
        print(f"\n  Total images downloaded: {total_images}")
    else:
        print("\n✗ No datasets downloaded")
    
    # Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review downloaded datasets")
    print("2. Organize images by breed")
    print("3. Selectively add to training data")
    print("4. Retrain model")
    
    print("\nDatasets are in: data/research_datasets/roboflow/")

if __name__ == "__main__":
    main()
