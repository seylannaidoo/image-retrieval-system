import pytest
import os
import yaml
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# We need to modify the path to find our modules
import sys

sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def test_config(tmp_path):
    """Create a temporary config for testing"""
    # Create config directories
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"

    # Create test config
    config_data = {
        "data_dir": str(data_dir),
        "model": {
            "name": "openai/clip-vit-base-patch32",
            "image_size": 224
        },
        "dataset": {
            "max_images": 2
        },
        "web": {
            "host": "localhost",
            "port": 5000,
            "max_results": 20
        }
    }

    # Write config file
    config_path = config_dir / "settings.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    # Import and create config after writing the file
    from config.config import Config
    config = Config(str(config_path))

    # Create necessary directories
    data_dir.mkdir(exist_ok=True)
    config.raw_data_dir.mkdir(exist_ok=True)
    config.processed_data_dir.mkdir(exist_ok=True)
    config.index_dir.mkdir(exist_ok=True)

    return config


@pytest.fixture
def sample_images(test_config):
    """Create sample test images"""
    image_paths = []

    # Create a few test images
    for i in range(2):
        img_path = test_config.raw_data_dir / f"test_image_{i}.jpg"
        img = Image.new('RGB', (224, 224), color=('red' if i == 0 else 'blue'))
        img.save(img_path)
        image_paths.append(img_path)

    yield image_paths

    # Cleanup
    for path in image_paths:
        if path.exists():
            path.unlink()


def test_find_images(test_config, sample_images, monkeypatch):
    """Test the find_images function"""
    # Import after config is set up
    from scripts.process_dataset import find_images
    # Patch the config in process_dataset
    import scripts.process_dataset as pd
    monkeypatch.setattr(pd, "config", test_config)

    found_images = find_images()
    assert len(found_images) == len(sample_images)
    assert all(img.suffix.lower() in {'.jpg', '.jpeg', '.png'} for img in found_images)


def test_compute_file_hash(sample_images):
    """Test hash computation consistency"""
    from scripts.process_dataset import compute_file_hash
    path = str(sample_images[0])
    hash1 = compute_file_hash(path)
    hash2 = compute_file_hash(path)
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hash length


def test_process_dataset(test_config, sample_images, monkeypatch):
    """Test the dataset processing pipeline"""
    # Mock the encoder to avoid loading CLIP model
    from unittest.mock import Mock
    mock_encoder = Mock()
    mock_encoder.dimension = 512
    # Generate random embeddings and ensure they're float32
    mock_embeddings = np.random.randn(1, 512).astype(np.float32)
    # Normalize the embeddings (CLIP embeddings should be normalized)
    mock_embeddings = mock_embeddings / np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
    mock_encoder.encode_images.return_value = mock_embeddings

    # Import and patch after config is set up
    import scripts.process_dataset as pd
    monkeypatch.setattr(pd, "CLIPEncoder", Mock(return_value=mock_encoder))
    monkeypatch.setattr(pd, "config", test_config)

    # Run processing
    pd.process_dataset()

    # Verify outputs
    assert (Path(test_config.index_dir) / "index.faiss").exists()
    assert (Path(test_config.index_dir) / "metadata.pkl").exists()
    assert (Path(test_config.data_dir) / "process_log.json").exists()