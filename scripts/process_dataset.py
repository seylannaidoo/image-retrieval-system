import os
import sys
from pathlib import Path
import hashlib
import json
from typing import List, Dict
import shutil
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models.clip_encoder import CLIPEncoder
from src.models.vector_store import VectorStore
from config.config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_URL = "https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/download"
DATASET_ZIP = "ai-vs-human-generated-dataset.zip"


def download_dataset():
    """
    Download dataset from Kaggle.
    Note: Requires Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        logger.info("Downloading dataset from Kaggle...")

        # Download dataset using Kaggle API
        kaggle.api.dataset_download_files(
            'alessandrasala79/ai-vs-human-generated-dataset',
            path=config.raw_data_dir,
            unzip=True
        )

        # Move files from test_data_v2 to raw_data_dir
        test_data_dir = config.raw_data_dir / 'test_data_v2'
        if test_data_dir.exists():
            for img_file in test_data_dir.glob('*.jpg'):
                shutil.move(str(img_file), str(config.raw_data_dir / img_file.name))
            test_data_dir.rmdir()

        logger.info("Dataset downloaded and extracted successfully")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("\nPlease ensure you have:")
        logger.error("1. Kaggle API credentials in ~/.kaggle/kaggle.json")
        logger.error("2. Install kaggle package: pip install kaggle")
        logger.error(
            "3. Or manually download from: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset")
        logger.error("   and place the images in the data/raw directory")
        sys.exit(1)


def setup_kaggle_credentials():
    """
    Helper to set up Kaggle API credentials if not present
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if not kaggle_json.exists():
        logger.info("Kaggle credentials not found. Please:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Create a new API token")
        logger.info("3. Download kaggle.json")
        logger.info("4. Place it in ~/.kaggle/kaggle.json")
        logger.info("5. Run this script again")
        sys.exit(1)

    # Ensure correct permissions
    os.chmod(kaggle_json, 0o600)


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def load_or_create_process_log() -> Dict:
    """Load or create processing log to track what's been done"""
    log_path = config.data_dir / 'process_log.json'
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return {
        'processed_images': {},
        'index_hash': None,
        'model_name': None
    }


def save_process_log(log: Dict) -> None:
    """Save processing log"""
    log_path = config.data_dir / 'process_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def find_images() -> List[Path]:
    """Find all image files in raw data directory"""
    extensions = {'.jpg', '.jpeg', '.png'}
    images = []
    for ext in extensions:
        images.extend(config.raw_data_dir.glob(f'*{ext}'))
    return sorted(images)[:config.max_images]


def process_dataset():
    """Main processing function"""
    logger.info("Starting dataset processing...")

    # Check if dataset exists
    if not any(config.raw_data_dir.glob('*.jpg')):
        logger.info("Dataset not found in raw directory")
        download_dataset()

    # Load or create processing log
    process_log = load_or_create_process_log()

    # Find images to process
    image_paths = find_images()
    logger.info(f"Found {len(image_paths)} images")

    # Initialize encoder
    encoder = CLIPEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

    # Check if we need to recompute embeddings
    recompute_needed = (
            process_log['model_name'] != config.model_name or
            len(process_log['processed_images']) != len(image_paths)
    )

    if not recompute_needed:
        print("Checking for changes in image files...")
        for img_path in image_paths:
            current_hash = compute_file_hash(str(img_path))
            stored_hash = process_log['processed_images'].get(str(img_path))
            if stored_hash != current_hash:
                recompute_needed = True
                break

    if not recompute_needed and (config.index_dir / "index.faiss").exists():
        print("No changes detected and index exists. Skipping processing.")
        return

    print("Processing images and computing embeddings...")

    # Process images and compute embeddings
    all_embeddings = []
    metadata = []

    for img_path in tqdm(image_paths):
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')

            # Store metadata
            metadata.append({
                'path': str(img_path),
                'filename': img_path.name
            })

            # Update process log
            process_log['processed_images'][str(img_path)] = compute_file_hash(str(img_path))

            # Get embedding
            embedding = encoder.encode_images([img])
            all_embeddings.append(embedding)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if not all_embeddings:
        print("No images were successfully processed!")
        return

    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)

    # Create and save vector store
    print("Creating vector index...")
    vector_store = VectorStore(dimension=encoder.dimension)
    vector_store.add(embeddings, metadata)
    vector_store.save(config.index_dir)

    # Update and save process log
    process_log['model_name'] = config.model_name
    save_process_log(process_log)

    print(f"Processing complete! {len(metadata)} images indexed.")


if __name__ == "__main__":
    process_dataset()