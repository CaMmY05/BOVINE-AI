"""
Bulk Image Downloader for Cattle Breeds
Downloads images from multiple sources automatically
"""

import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    import subprocess
    packages = [
        'bing-image-downloader',
        'icrawler',
        'requests',
        'beautifulsoup4'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.run(['pip', 'install', package], check=True, capture_output=True)
            print(f"✓ Installed {package}")
        except:
            print(f"✗ Failed to install {package}")

def download_from_bing(breed_name, queries, limit=200):
    """Download images from Bing"""
    try:
        from bing_image_downloader import downloader
        
        output_dir = f'data/raw_downloads/{breed_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Downloading {breed_name} images from Bing")
        print(f"{'='*60}")
        
        total_downloaded = 0
        for query in queries:
            print(f"\nQuery: {query}")
            try:
                downloader.download(
                    query,
                    limit=limit,
                    output_dir=output_dir,
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                    verbose=True
                )
                
                # Count downloaded images
                query_dir = Path(output_dir) / query
                if query_dir.exists():
                    count = len(list(query_dir.glob('*.jpg'))) + len(list(query_dir.glob('*.png')))
                    total_downloaded += count
                    print(f"✓ Downloaded {count} images for '{query}'")
            except Exception as e:
                print(f"✗ Error downloading '{query}': {e}")
        
        print(f"\n✓ Total downloaded for {breed_name}: {total_downloaded} images")
        return total_downloaded
        
    except ImportError:
        print("✗ bing-image-downloader not installed. Run: pip install bing-image-downloader")
        return 0

def download_from_google(breed_name, queries, limit=200):
    """Download images from Google using icrawler"""
    try:
        from icrawler.builtin import GoogleImageCrawler
        
        output_dir = f'data/raw_downloads/{breed_name}/google'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Downloading {breed_name} images from Google")
        print(f"{'='*60}")
        
        total_downloaded = 0
        for idx, query in enumerate(queries):
            print(f"\nQuery: {query}")
            try:
                google_crawler = GoogleImageCrawler(
                    storage={'root_dir': f'{output_dir}/query_{idx}'}
                )
                google_crawler.crawl(
                    keyword=query,
                    max_num=limit,
                    min_size=(200, 200),
                    max_size=None
                )
                
                # Count downloaded
                query_dir = Path(f'{output_dir}/query_{idx}')
                if query_dir.exists():
                    count = len(list(query_dir.glob('*.jpg'))) + len(list(query_dir.glob('*.png')))
                    total_downloaded += count
                    print(f"✓ Downloaded {count} images for '{query}'")
            except Exception as e:
                print(f"✗ Error downloading '{query}': {e}")
        
        print(f"\n✓ Total downloaded for {breed_name}: {total_downloaded} images")
        return total_downloaded
        
    except ImportError:
        print("✗ icrawler not installed. Run: pip install icrawler")
        return 0

def search_kaggle_datasets(keyword):
    """Search Kaggle for datasets"""
    import subprocess
    
    print(f"\n{'='*60}")
    print(f"Searching Kaggle for: {keyword}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '-s', keyword],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("\n💡 To download a dataset, use:")
        print("   kaggle datasets download -d <dataset-name>")
    except Exception as e:
        print(f"✗ Error searching Kaggle: {e}")
        print("💡 Make sure kaggle CLI is installed and configured")

def main():
    """Main download function"""
    
    print("="*60)
    print("CATTLE BREED IMAGE BULK DOWNLOADER")
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
    
    print("\nOptions:")
    print("1. Download from Bing (Recommended - Fast)")
    print("2. Download from Google (Slower but good quality)")
    print("3. Search Kaggle datasets")
    print("4. Download ALL (Bing + Google)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Download from Bing
        images_per_query = int(input("Images per query (default 100): ") or "100")
        
        for breed, queries in breeds_queries.items():
            download_from_bing(breed, queries, limit=images_per_query)
    
    elif choice == '2':
        # Download from Google
        images_per_query = int(input("Images per query (default 100): ") or "100")
        
        for breed, queries in breeds_queries.items():
            download_from_google(breed, queries, limit=images_per_query)
    
    elif choice == '3':
        # Search Kaggle
        search_kaggle_datasets('cattle breed')
        search_kaggle_datasets('indian cattle')
        search_kaggle_datasets('gir cattle')
        search_kaggle_datasets('sahiwal')
        search_kaggle_datasets('red sindhi')
    
    elif choice == '4':
        # Download from both
        images_per_query = int(input("Images per query (default 100): ") or "100")
        
        print("\n🚀 Starting bulk download from multiple sources...")
        print("This may take 30-60 minutes depending on your internet speed.")
        
        total_images = 0
        for breed, queries in breeds_queries.items():
            print(f"\n{'='*60}")
            print(f"Processing: {breed.upper()}")
            print(f"{'='*60}")
            
            bing_count = download_from_bing(breed, queries, limit=images_per_query)
            google_count = download_from_google(breed, queries, limit=images_per_query)
            
            breed_total = bing_count + google_count
            total_images += breed_total
            
            print(f"\n✓ {breed}: {breed_total} images downloaded")
        
        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE!")
        print(f"{'='*60}")
        print(f"Total images downloaded: {total_images}")
        print(f"\nNext steps:")
        print("1. Review images in data/raw_downloads/")
        print("2. Remove duplicates: python scripts/remove_duplicates.py")
        print("3. Move good images to data/raw/<breed_name>/")
        print("4. Run: python scripts/prepare_data.py")
    
    else:
        print("Exiting...")

if __name__ == "__main__":
    # Check and install requirements
    print("Checking requirements...")
    try:
        import bing_image_downloader
        import icrawler
    except ImportError:
        print("\n⚠️  Required packages not found!")
        install = input("Install required packages? (y/n): ").strip().lower()
        if install == 'y':
            install_requirements()
        else:
            print("Please install manually: pip install bing-image-downloader icrawler")
            exit(1)
    
    main()
