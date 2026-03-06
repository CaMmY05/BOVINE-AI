"""
Download additional high-quality Kaggle datasets from research
"""

import subprocess
import os
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_dir):
    """Download a Kaggle dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', output_dir, '--unzip'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Downloaded successfully to: {output_dir}")
        
        # Count files
        file_count = sum(1 for _ in Path(output_dir).rglob('*') if _.is_file())
        print(f"✓ Total files: {file_count}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading: {e}")
        print(f"  Output: {e.stdout}")
        print(f"  Error: {e.stderr}")
        return False

def main():
    """Download all recommended datasets from research"""
    
    print("="*60)
    print("DOWNLOADING HIGH-QUALITY KAGGLE DATASETS")
    print("From Parallel.ai Research Report")
    print("="*60)
    
    # Datasets to download (from research report)
    datasets = [
        {
            'name': 'sujayroy723/indian-cattle-breeds',
            'description': 'Indian Cattle Breeds - 5,949 images, balanced baseline',
            'output': 'data/kaggle_downloads/indian_cattle_breeds',
            'priority': 'HIGH'
        }
        # Note: lukex9442/indian-bovine-breeds already downloaded
    ]
    
    print("\nDatasets to download:")
    for idx, ds in enumerate(datasets, 1):
        print(f"{idx}. {ds['description']}")
        print(f"   Priority: {ds['priority']}")
    
    print("\n" + "="*60)
    confirm = input("Start download? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\n🚀 Starting downloads...\n")
    
    success_count = 0
    for ds in datasets:
        if download_kaggle_dataset(ds['name'], ds['output']):
            success_count += 1
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {success_count}/{len(datasets)}")
    print(f"\nAll datasets saved to: data/kaggle_downloads/")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Organize downloaded images")
    print("2. Merge with Google downloads")
    print("3. Remove duplicates")
    print("4. Retrain model")

if __name__ == "__main__":
    main()
