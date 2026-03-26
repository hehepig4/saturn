"""Embedding functions and registry."""

from .custom_embeddings import (
    LocalCLIPTextEmbedding,
    LocalCLIPImageEmbedding,
)
from .embedding_registry import (
    EmbeddingFunctionRegistry,
    get_registry,
    get_embedding_function,
    get_embedding_function_for_modality,
)
__all__ = [
    # Custom embedding implementations
    "LocalCLIPTextEmbedding",
    "LocalCLIPImageEmbedding",
    # Registry
    "EmbeddingFunctionRegistry",
    "get_registry",
    "get_embedding_function",
    "get_embedding_function_for_modality",
]
