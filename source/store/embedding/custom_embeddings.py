"""
Custom LanceDB Embedding Functions using Local Models.

This module provides custom embedding functions that use locally stored models
instead of downloading from the internet.

Supported models:
- CLIP (text and image)
- BGE-M3 (text, multilingual)
"""

from loguru import logger
from typing import List, Union, Optional, Any
from pathlib import Path
import threading

from core.paths import model_path
import torch
import numpy as np
from PIL import Image
from pydantic import Field, PrivateAttr
from lancedb.embeddings import EmbeddingFunction, register
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


# ===================== BGE-M3 Embedding =====================

class LocalBGEM3Embedding(EmbeddingFunction):
    """
    Custom LanceDB embedding function for text using local BGE-M3 model.
    
    BGE-M3 produces dense embeddings (1024 dim) suitable for semantic search.
    Uses SentenceTransformers backend for efficient batch processing.
    """
    
    model_path: str = Field(description="Path to local BGE-M3 model directory")
    device: str = Field(default="cuda", description="Device to run model on")
    batch_size: int = Field(default=128, description="Batch size for processing")
    max_length: int = Field(default=512, description="Maximum sequence length")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    
    _model: Any = PrivateAttr()
    _model_path_obj: Path = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)
    _lock: threading.Lock = PrivateAttr()
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 128,
        max_length: int = 512,
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            **kwargs
        )
        self._model_path_obj = Path(model_path)
        self._initialized = False
        self._model = None
        self._lock = threading.Lock()
    
    def _ensure_initialized(self):
        """Lazy initialization using SentenceTransformers for optimal performance.
        
        Thread-safe: Uses lock to prevent concurrent model loading.
        """
        # Fast path: return if already initialized (no lock needed)
        if self._initialized:
            return
        
        # Acquire lock for initialization
        with self._lock:
            # Double-check after acquiring lock (another thread may have initialized)
            if self._initialized:
                return
            
            logger.info(f"Loading BGE-M3 from {self._model_path_obj} (SentenceTransformers backend)")
            
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            
            # Determine the target GPU device
            device_str = self.device
            if device_str.startswith("cuda"):
                # Check if CUDA_VISIBLE_DEVICES is set
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                
                if visible_devices is None:
                    # CUDA_VISIBLE_DEVICES not set - explicitly select GPU 0
                    # This avoids PyTorch checking memory on all GPUs
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        torch.cuda.set_device(0)
                        device_str = "cuda:0"
                        logger.info(f"  Using GPU 0 (CUDA_VISIBLE_DEVICES not set, device_count={torch.cuda.device_count()})")
                else:
                    logger.info(f"  CUDA_VISIBLE_DEVICES={visible_devices}")
                    # Use the first visible device
                    device_str = "cuda:0"
            
            # Explicitly load to CPU first, then move to target device
            # This avoids meta tensor issues with concurrent access
            self._model = SentenceTransformer(
                str(self._model_path_obj),
                device="cpu",  # Load to CPU first
                trust_remote_code=True,
            )
            # Then move to target device
            if device_str != "cpu":
                self._model = self._model.to(device_str)
            
            self._initialized = True
            logger.info(f"✓ Loaded BGE-M3 text encoder (dim={self.ndims()}, FP32) on {device_str}")
    
    def ndims(self) -> int:
        """Return embedding dimension."""
        return 1024  # BGE-M3 always outputs 1024-dim dense vectors
    
    def compute_query_embeddings(
        self, query: str, *args, **kwargs
    ) -> List[np.ndarray]:
        """Compute embedding for a single query text."""
        return self.compute_source_embeddings([query], *args, **kwargs)
    
    def compute_source_embeddings(
        self, texts: List[str], *args, **kwargs
    ) -> List[np.ndarray]:
        """Compute embeddings for a list of texts using SentenceTransformers."""
        self._ensure_initialized()
        
        # SentenceTransformers handles batching internally and efficiently
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return list(embeddings)
    
    def source_field(self) -> str:
        """Return the name of the field to embed."""
        return "text"


def get_bge_m3_embedding(
    bge_model_path: str = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> LocalBGEM3Embedding:
    """
    Create a BGE-M3 text embedding function.
    
    Args:
        bge_model_path: Path to local BGE-M3 model
        device: Device to run model on
        batch_size: Batch size for processing
        
    Returns:
        LocalBGEM3Embedding instance
    """
    if bge_model_path is None:
        bge_model_path = str(model_path("bge-m3"))
    return LocalBGEM3Embedding(
        model_path=bge_model_path,
        device=device,
        batch_size=batch_size,
    )


# ===================== CLIP Embedding =====================

class LocalCLIPTextEmbedding(EmbeddingFunction):
    """
    Custom LanceDB embedding function for text using local CLIP model.
    
    This class wraps a locally stored CLIP model to generate text embeddings
    without downloading from the internet.
    """
    
    # Pydantic fields (must be declared at class level for Pydantic v2)
    model_path: str = Field(description="Path to local CLIP model directory")
    device: str = Field(default="cpu", description="Device to run model on")
    batch_size: int = Field(default=256, description="Batch size for processing (optimized for H100 80G)")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    
    # Private attributes (not serialized by Pydantic)
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model_path_obj: Path = PrivateAttr()
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 256,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize local CLIP text embedding function.
        
        Args:
            model_path: Path to local CLIP model directory
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
        """
        # Initialize Pydantic model
        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            **kwargs
        )
        
        # Set private attributes
        self._model_path_obj = Path(model_path)
        
        # Load model and tokenizer from local path
        logger.info(f"Loading CLIP model from {self._model_path_obj}")
        self._model = CLIPModel.from_pretrained(
            str(self._model_path_obj),
            local_files_only=True
        )
        self._tokenizer = CLIPTokenizer.from_pretrained(
            str(self._model_path_obj),
            local_files_only=True
        )
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"✓ Loaded CLIP text encoder from {self._model_path_obj}")
    
    def ndims(self) -> int:
        """Return embedding dimension."""
        return self._model.config.projection_dim
    
    def compute_query_embeddings(
        self, query: str, *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embedding for a single query text.
        
        Args:
            query: Input text string
            
        Returns:
            List containing single embedding array
        """
        return self.compute_source_embeddings([query], *args, **kwargs)
    
    def compute_source_embeddings(
        self, texts: List[str], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of embedding arrays
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize (CLIP tokenizer will use its default max_length)
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
                
                if self.normalize:
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                batch_embeddings = outputs.cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def source_field(self) -> str:
        """Return the name of the field to embed."""
        return "text"

class LocalCLIPImageEmbedding(EmbeddingFunction):
    """
    Custom LanceDB embedding function for images using local CLIP model.
    
    This class wraps a locally stored CLIP model to generate image embeddings
    without downloading from the internet.
    """
    
    # Pydantic fields (must be declared at class level for Pydantic v2)
    model_path: str = Field(description="Path to local CLIP model directory")
    device: str = Field(default="cpu", description="Device to run model on")
    batch_size: int = Field(default=256, description="Batch size for processing (optimized for H100 80G)")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    
    # Private attributes (not serialized by Pydantic)
    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _model_path_obj: Path = PrivateAttr()
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 256,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize local CLIP image embedding function.
        
        Args:
            model_path: Path to local CLIP model directory
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
        """
        # Initialize Pydantic model
        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            **kwargs
        )
        
        # Set private attributes
        self._model_path_obj = Path(model_path)
        
        # Load model and processor from local path
        logger.info(f"Loading CLIP model from {self._model_path_obj}")
        self._model = CLIPModel.from_pretrained(
            str(self._model_path_obj),
            local_files_only=True
        )
        self._processor = CLIPProcessor.from_pretrained(
            str(self._model_path_obj),
            local_files_only=True
        )
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"✓ Loaded CLIP image encoder from {self._model_path_obj}")
    
    def ndims(self) -> int:
        """Return embedding dimension."""
        return self._model.config.projection_dim
    
    def compute_query_embeddings(
        self, query: Union[str, bytes, Image.Image], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embedding for a single query image.
        
        Args:
            query: Input image (path, bytes, or PIL Image)
            
        Returns:
            List containing single embedding array
        """
        return self.compute_source_embeddings([query], *args, **kwargs)
    
    def compute_source_embeddings(
        self, images: List[Union[str, bytes, Image.Image]], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embeddings for a list of images.
        
        Args:
            images: List of images (paths, bytes, or PIL Images)
            
        Returns:
            List of embedding arrays
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Load and preprocess images
            pil_images = []
            for img in batch_images:
                try:
                    if isinstance(img, str):
                        # Load from path
                        pil_images.append(Image.open(img).convert("RGB"))
                    elif isinstance(img, bytes):
                        # Load from bytes
                        from io import BytesIO
                        pil_images.append(Image.open(BytesIO(img)).convert("RGB"))
                    elif isinstance(img, Image.Image):
                        # Already a PIL Image
                        pil_images.append(img.convert("RGB"))
                    else:
                        logger.warning(f"Unsupported image type: {type(img)}, using blank image")
                        # Use blank image as fallback
                        pil_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                except Exception as e:
                    logger.error(f"Failed to load image {img}: {e}, using blank image")
                    # Use blank image as fallback
                    pil_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            
            # Preprocess with CLIP processor
            inputs = self._processor(
                images=pil_images,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                
                if self.normalize:
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                batch_embeddings = outputs.cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def source_field(self) -> str:
        """Return the name of the field to embed."""
        return "image"
