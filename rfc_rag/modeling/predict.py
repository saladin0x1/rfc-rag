from pathlib import Path
import time

from loguru import logger
import typer

from rfc_rag.config import MODELS_DIR

app = typer.Typer()


@app.command()
def search(
    query: str,
    index_path: Path = MODELS_DIR / "rfc_faiss_index.bin",
    metadata_path: Path = MODELS_DIR / "rfc_metadata.pkl",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 5,
):
    """
    Search RFC chunks using semantic similarity.
    """
    # Fix for segfault on macOS by disabling multithreading in torch
    import torch
    torch.set_num_threads(1)
    
    import numpy as np
    import pickle
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Please install faiss-cpu and sentence-transformers")
        raise typer.Exit(1)

    if not index_path.exists() or not metadata_path.exists():
        logger.error(f"Index files not found. Run 'python -m rfc_rag.modeling.train' first.")
        raise typer.Exit(1)

    # Load FAISS index and metadata
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name)
    
    # Create query embedding
    logger.info(f"Searching for: '{query}'")
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    
    # Display results
    logger.info(f"\nTop {k} results for query: '{query}'")
    logger.info("=" * 80)
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # Valid result
            chunk = chunks[idx]
            logger.info(f"\nRank {i+1} | Similarity: {score:.4f}")
            logger.info(f"RFC: {chunk['rfc_id']} - {chunk['title']}")
            logger.info(f"Section: {chunk['section']}")
            logger.info(f"Text: {chunk['text'][:300]}...")
            logger.info("-" * 80)


@app.command()
def answer(
    query: str,
    index_path: Path = MODELS_DIR / "rfc_faiss_index.bin",
    metadata_path: Path = MODELS_DIR / "rfc_metadata.pkl",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 5,
    claude_model: str = "claude-3-haiku-20240307",
):
    """
    Answer questions using RFC chunks and Claude.
    """
    # Fix for segfault on macOS by disabling multithreading in torch
    import torch
    torch.set_num_threads(1)
    
    import numpy as np
    import pickle
    import os
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from anthropic import Anthropic
    except ImportError:
        logger.error("Please install faiss-cpu, sentence-transformers, and anthropic")
        raise typer.Exit(1)

    if not index_path.exists() or not metadata_path.exists():
        logger.error(f"Index files not found. Run 'python -m rfc_rag.modeling.train' first.")
        raise typer.Exit(1)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        raise typer.Exit(1)

    # Load FAISS index and metadata
    logger.info(f"Loading FAISS index from {index_path}")
    start_time = time.time()
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    load_time = time.time() - start_time
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    model_start = time.time()
    embed_model = SentenceTransformer(model_name)
    model_load_time = time.time() - model_start
    
    # Create query embedding and search
    logger.info(f"Searching for: '{query}'")
    embed_start = time.time()
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    embed_time = time.time() - embed_start
    
    search_start = time.time()
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    search_time = time.time() - search_start
    
    # Collect relevant chunks
    relevant_chunks = []
    logger.info(f"\nCollecting top {k} relevant chunks:")
    context_start = time.time()
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:
            chunk = chunks[idx]
            logger.info(f"  {i+1}. RFC {chunk['rfc_id']} - {chunk['title']}")
            logger.info(f"     Section: {chunk['section']} (similarity: {score:.4f})")
            logger.info(f"     Text preview: {chunk['text'][:100]}...")
            relevant_chunks.append({
                'rfc_id': chunk['rfc_id'],
                'title': chunk['title'],
                'section': chunk['section'],
                'text': chunk['text'],
                'score': float(score)
            })
    
    # Build context for Claude
    context = "\n\n".join([
        f"RFC {chunk['rfc_id']} - {chunk['title']}\nSection: {chunk['section']}\n{chunk['text']}"
        for chunk in relevant_chunks
    ])
    context_time = time.time() - context_start
    
    prompt = f"""Based on the following RFC documentation, please answer this question: {query}

Context from RFCs:
{context}

Please provide a clear, accurate answer based on the RFC content above. If the context doesn't contain enough information to fully answer the question, please say so."""

    # Show what's being sent to Claude
    logger.info(f"\nPrompt being sent to Claude:")
    logger.info("-" * 80)
    logger.info(f"Query: {query}")
    logger.info(f"Context length: {len(context)} characters")
    logger.info(f"Number of chunks: {len(relevant_chunks)}")
    logger.info("-" * 80)

    # Call Claude
    logger.info(f"Generating answer using {claude_model}...")
    client = Anthropic(api_key=api_key)
    
    claude_start = time.time()
    try:
        response = client.messages.create(
            model=claude_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        claude_time = time.time() - claude_start
        total_time = time.time() - start_time
        
        logger.info(f"\nClaude's Answer:")
        logger.info("=" * 80)
        logger.info(response.content[0].text)
        logger.info("=" * 80)
        
        # Show performance metrics
        logger.info(f"\nPerformance Metrics:")
        logger.info("-" * 80)
        logger.info(f"Index/metadata load time: {load_time:.3f}s")
        logger.info(f"Model load time: {model_load_time:.3f}s")
        logger.info(f"Query embedding time: {embed_time:.3f}s")
        logger.info(f"FAISS search time: {search_time:.3f}s")
        logger.info(f"Context building time: {context_time:.3f}s")
        logger.info(f"Claude API time: {claude_time:.3f}s")
        logger.info(f"Total end-to-end time: {total_time:.3f}s")
        logger.info("-" * 80)
        
        # Show relevance metrics
        avg_score = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)
        min_score = min(chunk['score'] for chunk in relevant_chunks)
        max_score = max(chunk['score'] for chunk in relevant_chunks)
        
        logger.info(f"\nRelevance Metrics:")
        logger.info("-" * 80)
        logger.info(f"Average similarity score: {avg_score:.4f}")
        logger.info(f"Min similarity score: {min_score:.4f}")
        logger.info(f"Max similarity score: {max_score:.4f}")
        logger.info(f"High relevance chunks (>0.7): {sum(1 for c in relevant_chunks if c['score'] > 0.7)}/{len(relevant_chunks)}")
        logger.info(f"Medium relevance chunks (0.5-0.7): {sum(1 for c in relevant_chunks if 0.5 <= c['score'] <= 0.7)}/{len(relevant_chunks)}")
        logger.info(f"Low relevance chunks (<0.5): {sum(1 for c in relevant_chunks if c['score'] < 0.5)}/{len(relevant_chunks)}")
        logger.info("-" * 80)
        
        # Also show the sources
        logger.info(f"\nSources (top {len(relevant_chunks)} matches):")
        for i, chunk in enumerate(relevant_chunks, 1):
            relevance_level = "HIGH" if chunk['score'] > 0.7 else "MED" if chunk['score'] > 0.5 else "LOW"
            logger.info(f"{i}. RFC {chunk['rfc_id']} - {chunk['title']}")
            logger.info(f"   Section: {chunk['section']} (similarity: {chunk['score']:.4f} - {relevance_level})")
        
    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
