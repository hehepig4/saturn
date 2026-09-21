"""
Embedding Configuration Adapter

Provides backward compatibility adapter that reads from embedding_models.json
and provides the same interface as the old embedding_config.py module.

This allows existing code to continue working while we transition to JSON-based configuration.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from .registry import get_registry, get_model_config

@dataclass
class EmbeddingFunctionConfig:
    """Configuration for a single embedding function."""
    
    # Identification
    name: str  # Unique name for this embedding function
    function_type: str  # Type: 'text' or 'image'
    
    # Model information
    model_name: str  # Name of the model (e.g., 'clip-vit-base-patch32')
    model_path: str  # Path to local model files
    
    # Processing parameters
    embedding_dim: int  # Dimension of output embeddings
    batch_size: int = 32  # Batch size for processing
    device: str = "cuda"  # Device to run on ('cpu' or 'cuda')
    normalize: bool = True  # Whether to normalize embeddings
    
    # Optional parameters
    description: str = ""  # Description of this function
    
    # Advanced options
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.name:
            return False, "name is required"
        
        if self.function_type not in ['text', 'image']:
            return False, f"Invalid function_type: {self.function_type}"
        
        if not self.model_path:
            return False, "model_path is required"
        
        if not Path(self.model_path).exists():
            return False, f"Model path does not exist: {self.model_path}"
        
        if self.embedding_dim <= 0:
            return False, f"Invalid embedding_dim: {self.embedding_dim}"
        
        return True, ""


def _load_embedding_functions_from_json() -> Dict[str, EmbeddingFunctionConfig]:
    """Load embedding functions from JSON configuration."""
    registry = get_registry()
    functions = {}
    
    # Load CLIP text and image functions
    clip_config = get_model_config("clip-vit-base-patch32")
    if clip_config:
        # CLIP Text
        functions["clip-text"] = EmbeddingFunctionConfig(
            name="clip-text",
            function_type="text",
            model_name="clip-vit-base-patch32",
            model_path=clip_config["local_path"],
            embedding_dim=clip_config["embedding_dim"],
            batch_size=clip_config["text_encoder"]["batch_size"],
            device="cuda",
            normalize=True,
            description=clip_config["description"] + " (text encoder)",
        )
        
        # CLIP Image
        functions["clip-image"] = EmbeddingFunctionConfig(
            name="clip-image",
            function_type="image",
            model_name="clip-vit-base-patch32",
            model_path=clip_config["local_path"],
            embedding_dim=clip_config["embedding_dim"],
            batch_size=clip_config["image_encoder"]["batch_size"],
            device="cuda",
            normalize=True,
            description=clip_config["description"] + " (image encoder)",
        )
    
    # Load BGE-M3 text function
    bge_config = get_model_config("bge-m3")
    if bge_config:
        functions["bge-m3"] = EmbeddingFunctionConfig(
            name="bge-m3",
            function_type="text",
            model_name="bge-m3",
            model_path=bge_config["local_path"],
            embedding_dim=bge_config["embedding_dim"],
            batch_size=bge_config["text_encoder"]["batch_size"],
            device="cuda",
            normalize=True,
            description=bge_config["description"],
            custom_params={
                "max_length": bge_config["text_encoder"].get("max_length", 8192),
                "use_fp16": bge_config["text_encoder"].get("use_fp16", True),
            }
        )
    
    return functions


# Load functions once at module import
EMBEDDING_FUNCTIONS: Dict[str, EmbeddingFunctionConfig] = _load_embedding_functions_from_json()

# Default mapping from modality to embedding function
DEFAULT_MODALITY_FUNCTIONS = {
    "TEXT": "bge-m3",
    "IMAGE": "clip-image",
}


def get_embedding_function_config(name: str) -> EmbeddingFunctionConfig:
    """Get embedding function configuration by name."""
    if name not in EMBEDDING_FUNCTIONS:
        raise KeyError(f"Embedding function '{name}' not found. Available: {list(EMBEDDING_FUNCTIONS.keys())}")
    return EMBEDDING_FUNCTIONS[name]


def get_default_function_for_modality(modality: str) -> str:
    """Get default embedding function name for a modality."""
    if modality not in DEFAULT_MODALITY_FUNCTIONS:
        raise KeyError(f"No default embedding function for modality: {modality}")
    return DEFAULT_MODALITY_FUNCTIONS[modality]


def validate_all_configs() -> Dict[str, tuple[bool, str]]:
    """Validate all embedding function configurations."""
    results = {}
    for name, config in EMBEDDING_FUNCTIONS.items():
        is_valid, error = config.validate()
        results[name] = (is_valid, error)
    return results


def get_function_summary() -> str:
    """Get a summary of all available embedding functions."""
    lines = ["Available Embedding Functions (from JSON config):", ""]
    
    for name, config in EMBEDDING_FUNCTIONS.items():
        lines.append(f"  • {name}:")
        lines.append(f"      Type: {config.function_type}")
        lines.append(f"      Model: {config.model_name}")
        lines.append(f"      Dimension: {config.embedding_dim}")
        lines.append(f"      Device: {config.device}")
        if config.description:
            lines.append(f"      Description: {config.description}")
        lines.append("")
    
    return "\n".join(lines)



