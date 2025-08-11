from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from rfc_rag.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command("embed")
def embed(
    input_path: Path = PROCESSED_DATA_DIR / "rfc_chunks.jsonl",
    output_path: Path = PROCESSED_DATA_DIR / "rfc_embeddings.npz",
    model: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
):
    """
    Generate embeddings for RFC chunks using local sentence-transformers.
    Fast, free, and no API rate limits.
    """
    import json
    import numpy as np
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Please install sentence-transformers: pip install sentence-transformers")
        raise typer.Exit(1)

    logger.info(f"Loading local embedding model: {model}")
    embed_model = SentenceTransformer(model)
    
    logger.info(f"Loading RFC chunks from {input_path}")
    chunks = []
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks.append(c)
            texts.append(c["text"])

    logger.info(f"Creating embeddings for {len(texts):,} chunks in batches of {batch_size}")
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Local embedding batches"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = embed_model.encode(
            batch_texts, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings, chunks=chunks)
    logger.success(
        f"Saved {len(chunks):,} embeddings to {output_path} | shape={embeddings.shape}"
    )


if __name__ == "__main__":
    app()
