"""
Automated bulk download - no interaction needed
Downloads 150 images per query from Bing and Google
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from bing_image_downloader import downloader
    from icrawler.builtin import GoogleImageCrawler
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(['pip', 'install', 'bing-image-downloader', 'icrawler'], check=True)
    from bing_image_downloader import downloader
    from icrawler.builtin import GoogleImageCrawler

# Define search queries
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

IMAGES_PER_QUERY = 150

print("="*60)
print("AUTOMATED BULK IMAGE DOWNLOADER")
print("="*60)
print(f"\nDownloading {IMAGES_PER_QUERY} images per query")
print("This will take 20-30 minutes...")
print("\n🚀 Starting download...\n")

total_images = 0

for breed, queries in breeds_queries.items():
    print(f"\n{'='*60}")
    print(f"Processing: {breed.upper()}")
    print(f"{'='*60}")
    
    breed_total = 0
    
    # Download from Bing
    print(f"\n📥 Downloading from Bing...")
    for query in queries:
        print(f"  Query: {query}")
        try:
            output_dir = f'data/raw_downloads/{breed}/bing'
            downloader.download(
                query,
                limit=IMAGES_PER_QUERY,
                output_dir=output_dir,
                adult_filter_off=True,
                force_replace=False,
                timeout=60,
                verbose=False
            )
            
            # Count downloaded
            query_dir = Path(output_dir) / query
            if query_dir.exists():
                count = len(list(query_dir.glob('*.jpg'))) + len(list(query_dir.glob('*.png')))
                breed_total += count
                print(f"    ✓ Downloaded {count} images")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Download from Google
    print(f"\n📥 Downloading from Google...")
    for idx, query in enumerate(queries):
        print(f"  Query: {query}")
        try:
            output_dir = f'data/raw_downloads/{breed}/google'
            google_crawler = GoogleImageCrawler(
                storage={'root_dir': f'{output_dir}/query_{idx}'}
            )
            google_crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(200, 200)
            )
            
            # Count downloaded
            query_dir = Path(f'{output_dir}/query_{idx}')
            if query_dir.exists():
                count = len(list(query_dir.glob('*.jpg'))) + len(list(query_dir.glob('*.png')))
                breed_total += count
                print(f"    ✓ Downloaded {count} images")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    total_images += breed_total
    print(f"\n✓ {breed}: {breed_total} images downloaded")

print(f"\n{'='*60}")
print(f"DOWNLOAD COMPLETE!")
print(f"{'='*60}")
print(f"Total images downloaded: {total_images}")
print(f"\nImages saved to: data/raw_downloads/")
print(f"\nNext steps:")
print(f"1. Remove duplicates: python scripts\\remove_duplicates.py")
print(f"2. Review and organize images")
print(f"3. Move to data/raw/<breed_name>/")
print(f"4. Retrain: python scripts\\train_classifier.py")
