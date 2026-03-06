"""
Simple Interactive Google Image Downloader
Works with Chrome - Just enter number of images and go!
"""

import os
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler

def download_google_images():
    """Download images from Google Images"""
    
    print("="*60)
    print("🌐 GOOGLE IMAGES DOWNLOADER (Chrome Compatible)")
    print("="*60)
    
    # Define search queries for each breed
    breeds_queries = {
        'gir': [
            'Gir cattle India',
            'Gir cow breed',
            'Gujarat Gir cattle',
            'Gir dairy cattle',
            'Gir bull India'
        ],
        'sahiwal': [
            'Sahiwal cattle',
            'Sahiwal cow breed',
            'Punjab Sahiwal cattle',
            'Sahiwal dairy cattle',
            'Sahiwal bull'
        ],
        'red_sindhi': [
            'Red Sindhi cattle',
            'Red Sindhi cow breed',
            'Lal Sindhi cattle',
            'Red Sindhi dairy cattle',
            'Sindh Red Sindhi cattle',
            'Red Sindhi bull'
        ]
    }
    
    print("\n📊 Queries per breed:")
    print(f"  - Gir: {len(breeds_queries['gir'])} queries")
    print(f"  - Sahiwal: {len(breeds_queries['sahiwal'])} queries")
    print(f"  - Red Sindhi: {len(breeds_queries['red_sindhi'])} queries (PRIORITY!)")
    
    # Get user input
    print("\n" + "="*60)
    images_per_query = input("How many images per query? (Recommended: 100-200): ").strip()
    
    try:
        images_per_query = int(images_per_query)
    except:
        print("Invalid input, using default: 150")
        images_per_query = 150
    
    print("\n" + "="*60)
    print(f"Configuration:")
    print(f"  - Images per query: {images_per_query}")
    print(f"  - Total queries: {sum(len(q) for q in breeds_queries.values())}")
    print(f"  - Expected total: ~{images_per_query * sum(len(q) for q in breeds_queries.values())} images")
    print("="*60)
    
    confirm = input("\nStart download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\n🚀 Starting download...")
    print("This will take 15-25 minutes depending on your internet speed.")
    print("You'll see progress for each query.\n")
    
    total_downloaded = 0
    
    # Download for each breed
    for breed, queries in breeds_queries.items():
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {breed.upper()}")
        print(f"{'='*60}")
        
        breed_count = 0
        
        for idx, query in enumerate(queries, 1):
            print(f"\n[{idx}/{len(queries)}] Query: {query}")
            
            try:
                output_dir = f'data/raw_downloads/{breed}/google_{idx}'
                os.makedirs(output_dir, exist_ok=True)
                
                # Create crawler
                google_crawler = GoogleImageCrawler(
                    storage={'root_dir': output_dir},
                    downloader_threads=4,  # Parallel downloads
                    log_level='ERROR'  # Only show errors
                )
                
                # Download images
                google_crawler.crawl(
                    keyword=query,
                    max_num=images_per_query,
                    min_size=(200, 200),  # Minimum image size
                    max_size=None
                )
                
                # Count downloaded
                downloaded = len(list(Path(output_dir).glob('*.jpg'))) + \
                           len(list(Path(output_dir).glob('*.png')))
                
                breed_count += downloaded
                print(f"    ✓ Downloaded {downloaded} images")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                print(f"    Continuing with next query...")
        
        total_downloaded += breed_count
        print(f"\n✓ {breed}: {breed_count} images downloaded")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Total images downloaded: {total_downloaded}")
    print(f"\nImages saved to: data/raw_downloads/")
    
    # Show breakdown
    print("\nBreakdown by breed:")
    for breed in breeds_queries.keys():
        breed_dir = Path(f'data/raw_downloads/{breed}')
        if breed_dir.exists():
            count = sum(1 for _ in breed_dir.rglob('*.jpg')) + \
                   sum(1 for _ in breed_dir.rglob('*.png'))
            print(f"  - {breed}: {count} images")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Remove duplicates:")
    print("   python scripts\\remove_duplicates.py")
    print("\n2. Review images (remove bad quality)")
    print("\n3. Move good images to data/raw/<breed_name>/")
    print("\n4. Retrain model:")
    print("   python scripts\\prepare_data.py")
    print("   python scripts\\extract_roi.py")
    print("   python scripts\\train_classifier.py")
    print("\n5. Evaluate:")
    print("   python scripts\\evaluate.py")
    print("="*60)

if __name__ == "__main__":
    try:
        from icrawler.builtin import GoogleImageCrawler
    except ImportError:
        print("❌ icrawler not installed!")
        print("Installing now...")
        import subprocess
        subprocess.run(['pip', 'install', 'icrawler'], check=True)
        print("✓ Installed! Please run the script again.")
        exit(0)
    
    download_google_images()
