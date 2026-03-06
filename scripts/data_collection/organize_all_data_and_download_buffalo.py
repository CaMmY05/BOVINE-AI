"""
Complete data organization script:
1. Organize downloaded Roboflow cow datasets by breed
2. Download buffalo breed datasets
3. Organize buffalo data by breed
4. Prepare final dataset structure for training
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import subprocess

def organize_roboflow_cow_data():
    """Organize Roboflow cow datasets by breed"""
    print("="*60)
    print("STEP 1: ORGANIZING ROBOFLOW COW DATA")
    print("="*60)
    
    # Target breeds for cows
    cow_breeds = ['gir', 'sahiwal', 'red_sindhi']
    
    # Source directories
    sources = [
        'data/research_datasets/roboflow/indian_bovine_recognition',
        'data/research_datasets/roboflow/kaggle_breed'
    ]
    
    # Destination
    dest_base = Path('data/research_datasets/organized/cows')
    
    breed_counts = defaultdict(int)
    
    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"⚠️  Source not found: {source}")
            continue
        
        print(f"\nProcessing: {source}")
        
        # Check for train/valid/test structure
        for split in ['train', 'valid', 'test']:
            split_dir = source_path / split
            if not split_dir.exists():
                continue
            
            print(f"  Processing {split} split...")
            
            # Look for breed folders or images with breed labels
            for item in split_dir.iterdir():
                if item.is_dir():
                    # Check if it's a breed folder
                    breed_name = item.name.lower().replace(' ', '_').replace('-', '_')
                    
                    # Map to our standard breed names
                    if 'gir' in breed_name:
                        breed = 'gir'
                    elif 'sahiwal' in breed_name:
                        breed = 'sahiwal'
                    elif 'red' in breed_name and 'sindhi' in breed_name:
                        breed = 'red_sindhi'
                    elif 'red_sindhi' in breed_name:
                        breed = 'red_sindhi'
                    else:
                        continue  # Skip unknown breeds
                    
                    # Copy images
                    dest_dir = dest_base / breed
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    for img_file in item.rglob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            dest_file = dest_dir / f"{source_path.name}_{split}_{img_file.name}"
                            if not dest_file.exists():
                                shutil.copy2(img_file, dest_file)
                                breed_counts[breed] += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("COW DATA ORGANIZED:")
    print(f"{'='*60}")
    total = 0
    for breed in cow_breeds:
        count = breed_counts[breed]
        print(f"  {breed:15s}: {count:5d} images")
        total += count
    print(f"  {'TOTAL':15s}: {total:5d} images")
    
    return breed_counts

def download_buffalo_datasets():
    """Download buffalo breed datasets from Roboflow"""
    print(f"\n{'='*60}")
    print("STEP 2: DOWNLOADING BUFFALO DATASETS")
    print(f"{'='*60}")
    
    # Buffalo breeds to target
    buffalo_breeds = ['murrah', 'jaffarabadi', 'mehsana', 'surti', 'bhadawari', 'nili_ravi']
    
    print("\nSearching for buffalo breed datasets on Roboflow...")
    print("Target breeds:", ', '.join(buffalo_breeds))
    
    # Known buffalo datasets (you may need to search Roboflow Universe for these)
    buffalo_datasets = [
        {
            'workspace': 'buffalo-detection',
            'project': 'murrah-buffalo',
            'version': 1,
            'description': 'Murrah Buffalo Detection',
            'breed': 'murrah'
        },
        # Add more as you find them on Roboflow Universe
    ]
    
    print("\n⚠️  Note: Buffalo datasets need to be identified on Roboflow Universe")
    print("Search at: https://universe.roboflow.com/")
    print("Keywords: 'buffalo', 'murrah', 'jaffarabadi', 'mehsana', etc.")
    
    # For now, create placeholder structure
    buffalo_base = Path('data/research_datasets/organized/buffaloes')
    for breed in buffalo_breeds:
        breed_dir = buffalo_base / breed
        breed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created buffalo breed folders in: {buffalo_base}")
    print("  Manual download required - see BUFFALO_DATASET_GUIDE.md")
    
    return buffalo_breeds

def check_kaggle_datasets():
    """Check if Kaggle dataset download is complete"""
    print(f"\n{'='*60}")
    print("STEP 3: CHECKING KAGGLE DATASETS")
    print(f"{'='*60}")
    
    kaggle_dir = Path('data/research_datasets/kaggle/indian_cattle_breeds')
    
    if kaggle_dir.exists():
        file_count = sum(1 for _ in kaggle_dir.rglob('*') if _.is_file())
        print(f"✓ Kaggle dataset found: {file_count} files")
        return True
    else:
        print("⏳ Kaggle dataset still downloading...")
        return False

def create_final_dataset_structure():
    """Create final organized dataset structure"""
    print(f"\n{'='*60}")
    print("STEP 4: CREATING FINAL DATASET STRUCTURE")
    print(f"{'='*60}")
    
    # Create structure
    base = Path('data/final_organized')
    
    # Cows
    cow_breeds = ['gir', 'sahiwal', 'red_sindhi']
    for breed in cow_breeds:
        (base / 'cows' / breed).mkdir(parents=True, exist_ok=True)
    
    # Buffaloes
    buffalo_breeds = ['murrah', 'jaffarabadi', 'mehsana']
    for breed in buffalo_breeds:
        (base / 'buffaloes' / breed).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created structure at: {base}")
    print("\nStructure:")
    print("  data/final_organized/")
    print("  ├── cows/")
    for breed in cow_breeds:
        print(f"  │   ├── {breed}/")
    print("  └── buffaloes/")
    for breed in buffalo_breeds:
        print(f"      ├── {breed}/")
    
    return base

def merge_with_original_data():
    """Merge organized data with original clean data"""
    print(f"\n{'='*60}")
    print("STEP 5: MERGING WITH ORIGINAL DATA")
    print(f"{'='*60}")
    
    # Original data (clean, working)
    original = Path('data/raw')
    organized = Path('data/research_datasets/organized/cows')
    final = Path('data/final_organized/cows')
    
    cow_breeds = ['gir', 'sahiwal', 'red_sindhi']
    
    for breed in cow_breeds:
        print(f"\nMerging {breed}...")
        
        # Count original
        original_dir = original / breed
        original_count = 0
        if original_dir.exists():
            original_count = len(list(original_dir.glob('*.jpg'))) + len(list(original_dir.glob('*.png')))
        
        # Count organized
        organized_dir = organized / breed
        organized_count = 0
        if organized_dir.exists():
            organized_count = len(list(organized_dir.glob('*.jpg'))) + len(list(organized_dir.glob('*.png')))
        
        # Copy to final
        final_dir = final / breed
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original (these are proven good)
        copied = 0
        if original_dir.exists():
            for img in original_dir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = final_dir / f"original_{img.name}"
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        copied += 1
        
        # Copy organized (new data - will review later)
        if organized_dir.exists():
            for img in organized_dir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = final_dir / f"new_{img.name}"
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        copied += 1
        
        final_count = len(list(final_dir.glob('*.jpg'))) + len(list(final_dir.glob('*.png')))
        
        print(f"  Original: {original_count}")
        print(f"  New: {organized_count}")
        print(f"  Final: {final_count}")

def main():
    """Main organization workflow"""
    print("="*60)
    print("COMPLETE DATA ORGANIZATION")
    print("Cows + Buffaloes")
    print("="*60)
    
    # Step 1: Organize Roboflow cow data
    cow_counts = organize_roboflow_cow_data()
    
    # Step 2: Download buffalo datasets
    buffalo_breeds = download_buffalo_datasets()
    
    # Step 3: Check Kaggle
    kaggle_ready = check_kaggle_datasets()
    
    # Step 4: Create final structure
    final_base = create_final_dataset_structure()
    
    # Step 5: Merge with original
    merge_with_original_data()
    
    # Summary
    print(f"\n{'='*60}")
    print("ORGANIZATION COMPLETE!")
    print(f"{'='*60}")
    
    print("\n📊 DATASET SUMMARY:")
    print("\nCows (Ready for Training):")
    print("  Location: data/final_organized/cows/")
    for breed, count in cow_counts.items():
        print(f"    {breed:15s}: ~{count} images (+ original)")
    
    print("\nBuffaloes (Need Manual Download):")
    print("  Location: data/final_organized/buffaloes/")
    print("  Breeds: murrah, jaffarabadi, mehsana")
    print("  Status: Folders created, awaiting data")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("\n1. Download buffalo datasets:")
    print("   - Search Roboflow Universe for buffalo breeds")
    print("   - Download manually to data/final_organized/buffaloes/")
    print("   - See: BUFFALO_DATASET_GUIDE.md")
    
    print("\n2. Review new cow images:")
    print("   - Check data/final_organized/cows/")
    print("   - Remove any poor quality images")
    print("   - Files prefixed with 'new_' are from Roboflow")
    
    print("\n3. Prepare for training:")
    print("   - Run: python scripts/prepare_data_v2.py")
    print("   - This will create train/val/test splits")
    
    print("\n4. Train models:")
    print("   - Cow model: python scripts/train_cow_classifier.py")
    print("   - Buffalo model: python scripts/train_buffalo_classifier.py")
    print("   - Combined: python scripts/train_combined_classifier.py")

if __name__ == "__main__":
    main()
