"""
Download buffalo breed images from Google
Modified from download_google_simple.py for buffalo breeds
"""

from icrawler.builtin import GoogleImageCrawler
import os
from pathlib import Path

def download_buffalo_images():
    """Download buffalo breed images"""
    
    print("="*60)
    print("BUFFALO BREED IMAGE DOWNLOADER")
    print("="*60)
    
    # Buffalo breeds and search queries
    BUFFALO_BREEDS = {
        'murrah': [
            'murrah buffalo india',
            'murrah buffalo side view',
            'murrah buffalo farm dairy',
            'murrah buffalo single animal',
            'murrah buffalo breed identification',
            'murrah buffalo haryana'
        ],
        'jaffarabadi': [
            'jaffarabadi buffalo',
            'jaffarabadi buffalo india',
            'jaffarabadi buffalo gujarat',
            'jaffarabadi buffalo side view',
            'jaffarabadi buffalo farm',
            'jaffarabadi buffalo breed'
        ],
        'mehsana': [
            'mehsana buffalo',
            'mehsana buffalo india',
            'mehsana buffalo gujarat',
            'mehsana buffalo side view',
            'mehsana buffalo farm',
            'mehsana buffalo breed'
        ]
    }
    
    # Output directory
    output_base = 'data/buffalo_downloads'
    
    print("\nTarget breeds:")
    for breed, queries in BUFFALO_BREEDS.items():
        print(f"  - {breed}: {len(queries)} queries")
    
    print(f"\nOutput directory: {output_base}")
    print("\nImages per query: 50-100")
    print("Total expected: 900-1,800 images")
    
    print("\n" + "="*60)
    confirm = input("Start download? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\n🚀 Starting downloads...\n")
    
    total_downloaded = 0
    breed_counts = {}
    
    for breed, queries in BUFFALO_BREEDS.items():
        print(f"\n{'='*60}")
        print(f"DOWNLOADING: {breed.upper()}")
        print(f"{'='*60}")
        
        breed_dir = os.path.join(output_base, breed)
        os.makedirs(breed_dir, exist_ok=True)
        
        breed_total = 0
        
        for idx, query in enumerate(queries, 1):
            print(f"\n[{idx}/{len(queries)}] Query: {query}")
            
            try:
                crawler = GoogleImageCrawler(
                    storage={'root_dir': breed_dir}
                )
                
                crawler.crawl(
                    keyword=query,
                    max_num=80,  # Download 80 per query
                    min_size=(200, 200),  # Minimum image size
                    file_idx_offset='auto'
                )
                
                # Count images
                current_count = len(list(Path(breed_dir).glob('*')))
                downloaded_this_query = current_count - breed_total
                breed_total = current_count
                
                print(f"  ✓ Downloaded {downloaded_this_query} images")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        breed_counts[breed] = breed_total
        total_downloaded += breed_total
        
        print(f"\n✓ {breed}: {breed_total} images downloaded")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Total images downloaded: {total_downloaded}")
    
    print("\nImages saved to:", output_base)
    print("\nBreakdown by breed:")
    for breed, count in breed_counts.items():
        print(f"  - {breed}: {count} images")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Remove duplicates:")
    print("   python scripts/remove_duplicates.py")
    print()
    print("2. Review images (remove bad quality)")
    print()
    print("3. Move good images to data/final_organized/buffaloes/<breed>/")
    print()
    print("4. Prepare data:")
    print("   python scripts/prepare_data_v2.py")
    print()
    print("5. Train buffalo model:")
    print("   python scripts/train_buffalo_classifier.py")
    print("="*60)

if __name__ == "__main__":
    try:
        from icrawler.builtin import GoogleImageCrawler
    except ImportError:
        print("✗ icrawler not installed!")
        print("Installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'icrawler'], check=True)
        from icrawler.builtin import GoogleImageCrawler
    
    download_buffalo_images()
