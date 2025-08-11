from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from rfc_rag.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    embeddings_path: Path = PROCESSED_DATA_DIR / "rfc_embeddings.npz",
    index_path: Path = MODELS_DIR / "rfc_faiss_index.bin",
    metadata_path: Path = MODELS_DIR / "rfc_metadata.pkl",
):
    """
    Build FAISS vector index from RFC embeddings for fast semantic search.
    """
    import numpy as np
    import pickle
    
    try:
        import faiss
    except ImportError:
        logger.error("Please install faiss-cpu: pip install faiss-cpu")
        raise typer.Exit(1)

    logger.info(f"Loading embeddings from {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data['embeddings']
    chunks = data['chunks']
    
    logger.info(f"Building FAISS index for {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]} dimensions)")
    
    # Create FAISS index for cosine similarity (inner product with normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings.astype(np.float32))
    
    # Save index and metadata
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(list(chunks), f)
    
    logger.success(f"Built FAISS index with {index.ntotal:,} vectors")
    logger.success(f"Saved index to {index_path}")
    logger.success(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    app()
