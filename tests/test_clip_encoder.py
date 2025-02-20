# tests/test_clip_encoder.py
import pytest
import torch
import numpy as np
from PIL import Image
from src.models.clip_encoder import CLIPEncoder


@pytest.fixture
def encoder():
    return CLIPEncoder(device="cpu")


@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    return img


def test_encoder_initialization(encoder):
    assert encoder.dimension > 0
    assert isinstance(encoder.model, torch.nn.Module)


def test_encode_images(encoder, sample_image):
    # Test single image
    embeddings = encoder.encode_images([sample_image])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == encoder.dimension

    # Test multiple images
    embeddings = encoder.encode_images([sample_image, sample_image])
    assert embeddings.shape == (2, encoder.dimension)


def test_encode_text(encoder):
    texts = ["a red square", "a blue circle"]
    embeddings = encoder.encode_text(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, encoder.dimension)


def test_embedding_normalization(encoder, sample_image):
    # Test if embeddings are normalized
    img_embedding = encoder.encode_images([sample_image])
    text_embedding = encoder.encode_text(["test"])

    # Check L2 norm ≈ 1
    assert np.abs(np.linalg.norm(img_embedding[0]) - 1.0) < 1e-5
    assert np.abs(np.linalg.norm(text_embedding[0]) - 1.0) < 1e-5