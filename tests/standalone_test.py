#!/usr/bin/env python3
"""
Standalone test runner for RFC search functionality.
Avoids threading conflicts that cause segfaults in pytest.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

def test_search_standalone():
    """Test search functionality without pytest threading issues."""
    
    # Set environment variables to prevent threading conflicts
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm globally
    
    try:
        import numpy as np
        import pickle
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Paths
    index_path = proj_root / "models" / "rfc_faiss_index.bin"
    metadata_path = proj_root / "models" / "rfc_metadata.pkl"
    
    print(f"🔍 Testing RFC search functionality...")
    print(f"📁 Index path: {index_path}")
    print(f"📁 Metadata path: {metadata_path}")
    
    # Check files exist
    if not index_path.exists():
        print(f"❌ FAISS index not found at {index_path}")
        return False
    if not metadata_path.exists():
        print(f"❌ Metadata not found at {metadata_path}")
        return False
    
    print("✅ Index files found")
    
    try:
        # Load FAISS index
        print("📚 Loading FAISS index...")
        index = faiss.read_index(str(index_path))
        print(f"✅ Index loaded: {index.ntotal} vectors")
        
        # Load metadata
        print("📋 Loading metadata...")
        with open(metadata_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✅ Metadata loaded: {len(chunks)} chunks")
        
        # Verify consistency
        assert index.ntotal == len(chunks), f"Index size {index.ntotal} != metadata size {len(chunks)}"
        print("✅ Index and metadata are consistent")
        
        # Load embedding model
        print("🤖 Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        print("✅ Embedding model loaded")
        
        # Test search
        query = "HTTP authentication methods"
        print(f"🔍 Searching for: '{query}'")
        
        # Create embedding
        query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype(np.float32), 3)
        
        # Verify results
        assert len(scores[0]) == 3, "Should return 3 results"
        assert len(indices[0]) == 3, "Should return 3 indices"
        
        valid_results = [idx for idx in indices[0] if idx != -1]
        assert len(valid_results) > 0, "Should have at least one valid result"
        
        print(f"✅ Found {len(valid_results)} valid results")
        
        # Display results
        print(f"\n🎯 Top {len(valid_results)} results:")
        print("=" * 60)
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                chunk = chunks[idx]
                print(f"\n📄 Rank {i+1} | Score: {score:.4f}")
                print(f"📋 RFC: {chunk['rfc_id']} - {chunk['title']}")
                print(f"📂 Section: {chunk['section']}")
                print(f"📝 Text: {chunk['text'][:150]}...")
                print("-" * 40)
        
        # Cleanup
        del embed_model
        del index
        
        print("\n🎉 All tests passed! RFC search functionality is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_search_standalone()
    sys.exit(0 if success else 1)
