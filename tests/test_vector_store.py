# tests/test_vector_store.py
import pytest
import numpy as np
from src.models.vector_store import VectorStore


@pytest.fixture
def vector_store():
    return VectorStore(dimension=512, distance_metric="cosine")


@pytest.fixture
def sample_vectors():
    # Create normalized random vectors
    vectors = np.random.randn(10, 512)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return vectors.astype(np.float32)


@pytest.fixture
def sample_metadata():
    return [{"path": f"test/image_{i}.jpg", "filename": f"image_{i}.jpg"} for i in range(10)]


def test_vector_store_initialization(vector_store):
    assert vector_store.dimension == 512
    assert vector_store.distance_metric == "cosine"
    assert len(vector_store) == 0


def test_adding_vectors(vector_store, sample_vectors, sample_metadata):
    vector_store.add(sample_vectors, sample_metadata)
    assert len(vector_store) == len(sample_vectors)


def test_search(vector_store, sample_vectors, sample_metadata):
    vector_store.add(sample_vectors, sample_metadata)

    # Search with first vector as query
    query = sample_vectors[0:1]
    distances, metadata = vector_store.search(query, k=3)

    assert len(distances) == 3
    assert len(metadata) == 3
    assert distances[0] > 0.99  # First result should be very similar to query


def test_save_load(vector_store, sample_vectors, sample_metadata, tmp_path):
    vector_store.add(sample_vectors, sample_metadata)

    # Save
    save_path = tmp_path / "vector_store"
    vector_store.save(save_path)

    # Load
    loaded_store = VectorStore.load(save_path)
    assert len(loaded_store) == len(vector_store)

    # Check search results match
    query = sample_vectors[0:1]
    original_distances, _ = vector_store.search(query, k=3)
    loaded_distances, _ = loaded_store.search(query, k=3)
    np.testing.assert_array_almost_equal(original_distances, loaded_distances)