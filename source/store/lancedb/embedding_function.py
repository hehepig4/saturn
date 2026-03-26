"""
LanceDB-compatible Embedding Function

Provides LanceDB-compatible embedding function wrapper for automatic vector generation.
"""

from lancedb.embeddings import EmbeddingFunction, register
from pydantic import Field
import numpy as np
from loguru import logger
from typing import Dict, Optional


@register("custom-clip")
class LanceDBEmbeddingFunction(EmbeddingFunction):
    """
    LanceDB-compatible embedding function that wraps our default_embed_function.
    
    This allows LanceDB to automatically generate embeddings during data insertion.
    
    Key: For images, expects binary data (bytes), not file paths!
    
    Usage:
        from lancedb.embeddings import get_registry
        func = get_registry().get("custom-clip").create(modality="IMAGE")
    """
    
    # Pydantic field for modality
    modality: str = Field(default="TEXT", description="Data modality (TEXT, IMAGE, etc.)")
    
    def ndims(self) -> int:
        """Return embedding dimension (fixed at 512 for CLIP/sentence-transformers)"""
        return 512
    
    def compute_source_embeddings(self, data):
        """
        Generate embeddings for a batch of data.
        
        Args:
            data: For TEXT - list of strings
                  For IMAGE - list of binary data (bytes)
            
        Returns:
            List of numpy arrays (embedding vectors)
        """
        try:
            # Import here to avoid circular imports
            from store.embedding import get_embedding_function_for_modality
            from PIL import Image
            import io
            
            # Get the appropriate embedding function for this modality
            embedding_func = get_embedding_function_for_modality(self.modality)
            
            if embedding_func is None:
                logger.error(f"No embedding function registered for modality: {self.modality}")
                # Return list of zero vectors as numpy arrays
                return [np.zeros(self.ndims(), dtype=np.float32) for _ in data]
            
            # Generate embeddings based on modality
            if self.modality == "TEXT":
                # Text: data is list of strings or list of list of strings
                # LanceDB passes PyArrow arrays, convert to Python list
                if hasattr(data, 'to_pylist'):
                    # PyArrow StringArray -> Python list
                    text_data = data.to_pylist()
                elif isinstance(data, (list, tuple)):
                    text_data = list(data)
                else:
                    text_data = [str(data)]
                
                embeddings_result = embedding_func.compute_source_embeddings(text_data)
            
            elif self.modality == "IMAGE":
                # Image: data is list of binary bytes
                # LanceDB passes PyArrow arrays, convert to Python list
                if hasattr(data, 'to_pylist'):
                    # PyArrow BinaryArray -> Python list
                    image_data = data.to_pylist()
                elif isinstance(data, (list, tuple)):
                    image_data = list(data)
                else:
                    image_data = [data]
                
                # Convert bytes to PIL Images for the embedding function
                images = []
                for img_bytes in image_data:
                    if img_bytes is None:
                        logger.warning("Encountered None image bytes, using zero embedding")
                        images.append(None)
                    else:
                        try:
                            # Open image from bytes
                            img = Image.open(io.BytesIO(img_bytes))
                            images.append(img)
                        except Exception as e:
                            logger.error(f"Failed to load image from bytes: {e}")
                            images.append(None)
                
                # Generate embeddings for images
                embeddings_result = embedding_func.compute_source_embeddings(images)
            
            else:
                # Fallback for unsupported modalities
                logger.warning(f"Unsupported modality: {self.modality}, using zero embeddings")
                return [np.zeros(self.ndims(), dtype=np.float32) for _ in data]
            
            # Convert all embeddings to numpy arrays
            embeddings = []
            for emb in embeddings_result:
                if emb is None:
                    # Use zero vector for failed embeddings
                    embeddings.append(np.zeros(self.ndims(), dtype=np.float32))
                elif isinstance(emb, np.ndarray):
                    embeddings.append(emb.astype(np.float32))
                else:
                    # Convert list to numpy array
                    embeddings.append(np.array(emb, dtype=np.float32))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed for {self.modality}: {e}")
            logger.exception("Detailed error:")
            # Fallback to zero embeddings as numpy arrays
            return [np.zeros(self.ndims(), dtype=np.float32) for _ in data]
    
    def compute_query_embeddings(self, data):
        """
        Generate embeddings for query data (same as source embeddings for our use case).
        
        Args:
            data: Query data (same format as source data)
            
        Returns:
            List of numpy arrays (embedding vectors)
        """
        return self.compute_source_embeddings(data)

