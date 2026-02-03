"""Embedding model loader"""
from typing import Optional
import os


def load_embedding_model(model_name: Optional[str] = None):
    """
    Load embedding model for semantic similarity calculations
    
    Args:
        model_name: Name of the model to load (default: from env or 'all-MiniLM-L6-v2')
    
    Returns:
        Loaded embedding model (sentence-transformers model)
    
    Example:
        model = load_embedding_model()
        embeddings = model.encode(["text1", "text2"])
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding-based evaluation. "
            "Install it with: pip install sentence-transformers"
        )
    
    # Get model name from environment or use default
    if model_name is None:
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"  # Fast and lightweight default
        )
    
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {e}"
        ) from e
