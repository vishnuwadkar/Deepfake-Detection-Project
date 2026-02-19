import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
DATASET_SLUG = "ciplab/real-and-fake-face-detection"
DATA_ROOT = Path("data/raw")
REAL_DIR = DATA_ROOT / "real"
FAKE_DIR = DATA_ROOT / "fake"

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_kaggle_setup():
    """Checks if kaggle is installed and configured."""
    # Check for API credentials first!
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    local_kaggle_json = Path("kaggle.json")

    if not kaggle_json.exists():
        if local_kaggle_json.exists():
            print(f"Found kaggle.json in current directory. Moving to {kaggle_dir}...")
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(local_kaggle_json), str(kaggle_json))
        else:
            print(f"\n[WARNING] kaggle.json not found in {kaggle_dir} or current directory.")
            print("The script might fail if you haven't authenticated.")
            # We don't exit yet, we let import kaggle try and fail if it must, 
            # or maybe the user has env vars set.

    try:
        import kaggle
    except ImportError:
        print("Kaggle library not found. Installing...")
        install_package("kaggle")
        print("Kaggle library installed.")
    except Exception as e:
        print(f"Error importing kaggle: {e}")
        # Continue and try to run command line anyway?
        pass
            
    # Set permissions for linux/mac if needed (os specific, skipping for windows mostly)
    # But we are on Windows per user info, so strictly file permissions aren't as strict as chmod 600

def download_and_organize():
    """Downloads the dataset and organizes it into real/fake folders."""
    print(f"Downloading dataset: {DATASET_SLUG}...")
    
    # Use kaggle command line to download
    # We use subprocess to avoid complex API handling if env vars aren't picked up immediately after move
    cmd = f"kaggle datasets download -d {DATASET_SLUG} -p {DATA_ROOT} --unzip"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Ensure you have accepted the dataset rules if required.")
        sys.exit(1)

    print("Download complete. Organizing files...")
    
    # Structure of ciplab/real-and-fake-face-detection after unzip:
    # data/raw/
    #   real_and_fake_face/
    #     real_and_fake_face/ (sometimes nested)
    #       training_real/
    #       training_fake/
    
    # Find the extraction root (it might be nested)
    # We look for 'training_real' to find the anchor
    
    found_root = None
    for path in DATA_ROOT.rglob("training_real"):
        if path.is_dir():
            found_root = path.parent
            break
            
    if not found_root:
        print("Could not find 'training_real' folder. Check download content.")
        return

    # Create destination directories
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Move files
    source_real = found_root / "training_real"
    source_fake = found_root / "training_fake"

    print(f"Moving real images from {source_real} to {REAL_DIR}...")
    for img in source_real.glob("*"):
        shutil.move(str(img), str(REAL_DIR / img.name))

    print(f"Moving fake images from {source_fake} to {FAKE_DIR}...")
    for img in source_fake.glob("*"):
        shutil.move(str(img), str(FAKE_DIR / img.name))

    # Cleanup source directories
    # Only remove the top-level folder that was extracted (e.g., real_and_fake_face)
    # Be careful not to delete data/raw/real or data/raw/fake
    
    # The zip usually extracts a folder named 'real_and_fake_face' or similar in data/raw
    # We can identify it by listing dirs in data/raw that are NOT real or fake
    for item in DATA_ROOT.iterdir():
        if item.is_dir() and item.name not in ["real", "fake"]:
            print(f"Cleaning up {item}...")
            shutil.rmtree(item)

    print("\nDataset setup complete!")
    print(f"Real images: {len(list(REAL_DIR.glob('*')))}")
    print(f"Fake images: {len(list(FAKE_DIR.glob('*')))}")

if __name__ == "__main__":
    check_kaggle_setup()
    download_and_organize()
