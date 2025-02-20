from typing import List, Union
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from .base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):
    """
    CLIP model wrapper for generating embeddings for images and text.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize the CLIP encoder.

        Args:
            model_name (str): Name of the CLIP model to use from HuggingFace
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Set model to evaluation mode
        self.model.eval()

    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of images.

        Args:
            images: List of image paths or PIL Image objects
            batch_size: Number of images to process at once

        Returns:
            numpy.ndarray: Image embeddings of shape (n_images, dimension)
        """
        all_embeddings = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load images if paths are provided
            batch_images = []
            for img in batch:
                if isinstance(img, str):
                    batch_images.append(Image.open(img).convert('RGB'))
                else:
                    batch_images.append(img)

            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_embeddings = self.model.get_image_features(**inputs)

            # Normalize embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of text queries.

        Args:
            texts: List of text queries
            batch_size: Number of texts to process at once

        Returns:
            numpy.ndarray: Text embeddings of shape (n_texts, dimension)
        """
        all_embeddings = []

        # Process text in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_embeddings = self.model.get_text_features(**inputs)

            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.config.projection_dim