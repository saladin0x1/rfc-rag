#!/usr/bin/env python3
"""
Test script for the RFC RAG search functionality.
"""

import sys
from pathlib import Path
import pytest

# Add the project root to Python path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))


def test_search_functionality():
    """Test that the RFC search functionality works correctly."""
    pytest.importorskip("faiss", reason="FAISS not available")
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not available")
    
    import numpy as np
    import pickle
    import faiss
    import os
    
    # Disable tqdm and minimize verbosity to prevent threading conflicts
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from sentence_transformers import SentenceTransformer
    
    # Paths
    index_path = proj_root / "models" / "rfc_faiss_index.bin"
    metadata_path = proj_root / "models" / "rfc_metadata.pkl"
    
    # Check that required files exist
    assert index_path.exists(), f"FAISS index not found at {index_path}"
    assert metadata_path.exists(), f"Metadata not found at {metadata_path}"
    
    # Load FAISS index and metadata
    index = faiss.read_index(str(index_path))
    assert index.ntotal > 0, "Index should contain vectors"
    
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    assert len(chunks) > 0, "Should have metadata chunks"
    assert index.ntotal == len(chunks), "Index size should match metadata size"
    
    # Load model with minimal verbosity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    # Test query
    query = "HTTP authentication methods"
    
    # Create embedding without progress bar
    query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype(np.float32), 3)
    
    # Verify results
    assert len(scores[0]) == 3, "Should return 3 results"
    assert len(indices[0]) == 3, "Should return 3 indices"
    
    # Check that we got valid results
    valid_results = [idx for idx in indices[0] if idx != -1]
    assert len(valid_results) > 0, "Should have at least one valid result"
    
    # Check similarity scores are reasonable (between 0 and 1)
    for score in scores[0]:
        if score > 0:  # Valid score
            assert 0 <= score <= 1, f"Similarity score {score} should be between 0 and 1"
    
    # Verify we can access the chunks
    for idx in valid_results:
        chunk = chunks[idx]
        assert 'rfc_id' in chunk, "Chunk should have rfc_id"
        assert 'title' in chunk, "Chunk should have title"
        assert 'section' in chunk, "Chunk should have section"
        assert 'text' in chunk, "Chunk should have text"
        assert len(chunk['text']) > 0, "Chunk text should not be empty"
    
    # Explicit cleanup to prevent threading issues
    del embed_model
    del index


def test_search_integration():
    """Integration test that runs a search and prints results."""
    pytest.importorskip("faiss", reason="FAISS not available")
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not available")
    
    import numpy as np
    import pickle
    import faiss
    import os
    
    # Disable tqdm to prevent threading conflicts
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from sentence_transformers import SentenceTransformer
    
    # Paths
    index_path = proj_root / "models" / "rfc_faiss_index.bin"
    metadata_path = proj_root / "models" / "rfc_metadata.pkl"
    
    if not (index_path.exists() and metadata_path.exists()):
        pytest.skip("Index files not found - run training first")
    
    # Load FAISS index and metadata
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load model with minimal verbosity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    # Test query
    query = "HTTP authentication methods"
    
    # Create embedding
    query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype(np.float32), 3)
    
    print(f"\n🔍 Top 3 results for: '{query}'")
    print("=" * 60)
    
    results_found = 0
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:
            chunk = chunks[idx]
            print(f"\nRank {i+1} | Score: {score:.4f}")
            print(f"RFC: {chunk['rfc_id']} - {chunk['title']}")
            print(f"Section: {chunk['section']}")
            print(f"Text: {chunk['text'][:200]}...")
            print("-" * 40)
            results_found += 1
    
    assert results_found > 0, "Should find at least one result"
    print(f"\n✅ Search integration test completed successfully with {results_found} results!")
    
    # Explicit cleanup to prevent threading issues
    del embed_model
    del index
