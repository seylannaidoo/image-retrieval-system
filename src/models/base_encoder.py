from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from PIL import Image


class BaseEncoder(ABC):
    """Base class for embedding models"""

    @abstractmethod
    def encode_images(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Encode images to embedding vectors

        Args:
            images: List of image paths or PIL Images

        Returns:
            Array of embeddings with shape (n_images, dimension)
        """
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text queries to embedding vectors

        Args:
            texts: List of text queries

        Returns:
            Array of embeddings with shape (n_texts, dimension)
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension"""
        pass