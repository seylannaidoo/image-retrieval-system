from typing import List, Tuple
import numpy as np
import faiss
import pickle
from pathlib import Path


class VectorStore:
    """Vector store for efficient similarity search of embeddings"""

    def __init__(self, dimension: int, distance_metric: str = "cosine"):
        """
        Initialize vector store

        Args:
            dimension: Dimensionality of vectors to store
            distance_metric: Either 'cosine' or 'l2'
        """
        self.dimension = dimension
        self.distance_metric = distance_metric

        if distance_metric == "cosine":
            # For cosine similarity, normalize vectors and use L2
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = True
        elif distance_metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        self.metadata = {}

    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """
        Add vectors and metadata to the index

        Args:
            vectors: Array of shape (n_vectors, dimension)
            metadata: List of metadata dicts for each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors and metadata entries must match")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")

        # Normalize if using cosine similarity
        if self.normalize:
            faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors.astype(np.float32))

        # Store metadata
        start_id = len(self.metadata)
        for i, meta in enumerate(metadata):
            self.metadata[start_id + i] = meta

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[dict]]:
        """
        Search for k nearest neighbors

        Args:
            query: Query vector of shape (1, dimension)
            k: Number of results to return

        Returns:
            Tuple of (distances, metadata)
        """
        if k > len(self.metadata):
            k = len(self.metadata)

        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.normalize:
            faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query.astype(np.float32), k)

        # Get metadata
        metadata = [self.metadata[int(idx)] for idx in indices[0]]

        # Convert L2 distances to cosine similarity if needed
        if self.normalize:
            distances = 1 - distances / 2

        return distances[0], metadata

    def save(self, path: str) -> None:
        """Save index and metadata to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata and config
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "dimension": self.dimension,
                "distance_metric": self.distance_metric
            }, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load index and metadata from disk"""
        path = Path(path)

        # Load metadata and config
        with open(path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        # Create instance
        instance = cls(
            dimension=data["dimension"],
            distance_metric=data["distance_metric"]
        )

        # Load index and metadata
        instance.index = faiss.read_index(str(path / "index.faiss"))
        instance.metadata = data["metadata"]

        return instance

    def __len__(self) -> int:
        return len(self.metadata)