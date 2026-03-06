"""
Process all downloaded images:
1. Remove duplicates
2. Organize into breed folders
3. Merge with existing data
4. Prepare for training
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm

def find_and_remove_duplicates(source_dir, threshold=5):
    """Find and remove duplicate images"""
    print(f"\n{'='*60}")
    print("STEP 1: REMOVING DUPLICATES")
    print(f"{'='*60}")
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(source_dir).rglob(ext))
    
    print(f"Found {len(image_files)} images")
    
    # Calculate hashes
    print("\nCalculating image hashes...")
    hashes = {}
    failed = []
    
    for img_path in tqdm(image_files):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_hash = imagehash.phash(img, hash_size=8)
                hashes[str(img_path)] = img_hash
        except Exception as e:
            failed.append(str(img_path))
    
    if failed:
        print(f"\n⚠️  Failed to process {len(failed)} images (will be removed)")
        for path in failed:
            try:
                os.remove(path)
            except:
                pass
    
    # Find duplicates
    print("\nFinding duplicates...")
    duplicates = defaultdict(list)
    processed = set()
    
    hash_list = list(hashes.items())
    for i, (path1, hash1) in enumerate(tqdm(hash_list)):
        if path1 in processed:
            continue
        
        group = [path1]
        for path2, hash2 in hash_list[i+1:]:
            if path2 in processed:
                continue
            
            distance = hash1 - hash2
            if distance <= threshold:
                group.append(path2)
                processed.add(path2)
        
        if len(group) > 1:
            duplicates[hash1].extend(group)
            processed.add(path1)
    
    # Remove duplicates (keep first)
    removed_count = 0
    if duplicates:
        print(f"\nFound {len(duplicates)} groups of duplicates")
        for group in tqdm(duplicates.values(), desc="Removing duplicates"):
            for img_path in group[1:]:
                try:
                    os.remove(img_path)
                    removed_count += 1
                except:
                    pass
    
    print(f"\n✓ Removed {removed_count} duplicate images")
    return removed_count

def organize_images(source_dir, dest_base='data/raw'):
    """Organize images into breed folders"""
    print(f"\n{'='*60}")
    print("STEP 2: ORGANIZING IMAGES")
    print(f"{'='*60}")
    
    breeds = ['gir', 'sahiwal', 'red_sindhi']
    organized_count = {}
    
    for breed in breeds:
        print(f"\nProcessing {breed}...")
        
        # Source directories
        source_breed_dir = Path(source_dir) / breed
        if not source_breed_dir.exists():
            print(f"  ⚠️  No directory found for {breed}")
            continue
        
        # Destination
        dest_dir = Path(dest_base) / breed
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing count
        existing_count = len(list(dest_dir.glob('*.jpg'))) + len(list(dest_dir.glob('*.png')))
        
        # Copy new images
        new_count = 0
        for img_file in source_breed_dir.rglob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Generate unique filename
                dest_file = dest_dir / f"download_{new_count:04d}{img_file.suffix}"
                while dest_file.exists():
                    new_count += 1
                    dest_file = dest_dir / f"download_{new_count:04d}{img_file.suffix}"
                
                try:
                    shutil.copy2(img_file, dest_file)
                    new_count += 1
                except Exception as e:
                    print(f"  ✗ Error copying {img_file.name}: {e}")
        
        total_count = existing_count + new_count
        organized_count[breed] = {'existing': existing_count, 'new': new_count, 'total': total_count}
        
        print(f"  ✓ {breed}:")
        print(f"    - Existing: {existing_count}")
        print(f"    - New: {new_count}")
        print(f"    - Total: {total_count}")
    
    return organized_count

def main():
    """Main processing pipeline"""
    print("="*60)
    print("PROCESSING ALL DOWNLOADED IMAGES")
    print("="*60)
    
    source_dir = 'data/raw_downloads'
    
    if not os.path.exists(source_dir):
        print(f"✗ Source directory not found: {source_dir}")
        return
    
    # Step 1: Remove duplicates
    removed = find_and_remove_duplicates(source_dir, threshold=5)
    
    # Step 2: Organize images
    counts = organize_images(source_dir, dest_base='data/raw')
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDuplicates removed: {removed}")
    print(f"\nFinal dataset:")
    total_new = 0
    total_all = 0
    for breed, data in counts.items():
        print(f"  {breed:15s}: {data['total']:4d} images (+{data['new']} new)")
        total_new += data['new']
        total_all += data['total']
    
    print(f"\n  {'TOTAL':15s}: {total_all:4d} images (+{total_new} new)")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Run data preparation:")
    print("   python scripts\\prepare_data.py")
    print("\n2. Extract ROIs:")
    print("   python scripts\\extract_roi.py")
    print("\n3. Train model:")
    print("   python scripts\\train_classifier.py")
    print("\n4. Evaluate:")
    print("   python scripts\\evaluate.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        import imagehash
    except ImportError:
        print("✗ imagehash not installed!")
        print("Installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'imagehash'], check=True)
        import imagehash
    
    main()
