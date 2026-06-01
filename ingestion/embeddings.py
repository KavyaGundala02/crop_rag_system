# ingestion/embeddings.py

from sentence_transformers import SentenceTransformer
import time

# ─────────────────────────────────────────
# Load Embedding Model
# ─────────────────────────────────────────

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence-transformer embedding model.
    
    We compare 2 models as required by assignment:
    
    Model 1: all-MiniLM-L6-v2
    - Fast, lightweight
    - Vector size: 384
    - Good for general sentences
    
    Model 2: all-mpnet-base-v2
    - Slower, more accurate
    - Vector size: 768
    - Better semantic understanding
    """
    
    print(f"\n Loading embedding model: {model_name}")
    print(" (First time will download the model — please wait...)")
    
    start = time.time()
    model = SentenceTransformer(model_name)
    end = time.time()
    
    print(f"✅ Model loaded in {round(end - start, 2)} seconds")
    return model


# ─────────────────────────────────────────
# Generate Embeddings
# ─────────────────────────────────────────

def generate_embeddings(model, chunks: list, batch_size: int = 64) -> list:
    """
    Generate embeddings for all chunks.
    Returns list of embeddings in same order as chunks.
    """
    
    print(f"\n Generating embeddings for {len(chunks)} chunks...")
    print(f" Batch size: {batch_size}")
    
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size  = batch_size,
        show_progress_bar = True
    )
    embeddings = embeddings.tolist()
    
    end = time.time()
    
    print(f"✅ Embeddings generated in {round(end - start, 2)} seconds")
    print(f" Embedding vector size: {len(embeddings[0])}")
    
    return embeddings


# ─────────────────────────────────────────
# Compare Two Models
# ─────────────────────────────────────────

def compare_models(chunks: list):
    """
    Compare all-MiniLM-L6-v2 vs all-mpnet-base-v2.
    Shows speed and vector size difference.
    """
    
    models_to_compare = [
        "all-MiniLM-L6-v2",   # Model 1 - fast
        "all-mpnet-base-v2"    # Model 2 - accurate
    ]
    
    # Use only first 100 chunks for comparison (faster)
    sample_chunks = chunks[:100]
    
    print("\n" + "=" * 60)
    print(" MODEL COMPARISON (on 100 sample chunks)")
    print("=" * 60)
    
    results = {}
    
    for model_name in models_to_compare:
        model = load_embedding_model(model_name)
        
        start = time.time()
        embeddings = generate_embeddings(model, sample_chunks, batch_size=32)
        end = time.time()
        
        results[model_name] = {
            "time"        : round(end - start, 2),
            "vector_size" : len(embeddings[0])
        }
        
        print(f"\n Model    : {model_name}")
        print(f" Time     : {results[model_name]['time']} seconds")
        print(f" Vec Size : {results[model_name]['vector_size']}")
    
    print("\n" + "=" * 60)
    print(" VERDICT")
    print("=" * 60)
    print(" all-MiniLM-L6-v2  → faster, lighter, good enough for this task")
    print(" all-mpnet-base-v2 → slower, larger vectors, better accuracy")
    print("\n✅ We will use: all-MiniLM-L6-v2 (best speed/accuracy tradeoff)")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from load_data import load_dataset
    from chunking import row_level_chunks, crop_aggregate_chunks
    
    # Load data
    df = load_dataset("data/Crop_recommendation.csv")
    
    # Get both chunk types
    row_chunks = row_level_chunks(df)
    agg_chunks = crop_aggregate_chunks(df)
    
    # Combine for hybrid strategy
    all_chunks = row_chunks + agg_chunks
    print(f"\n Total hybrid chunks: {len(all_chunks)}")
    
    # Compare models first
    compare_models(row_chunks)
    
    # Generate final embeddings using chosen model
    print("\n" + "=" * 60)
    print(" GENERATING FINAL EMBEDDINGS (all hybrid chunks)")
    print("=" * 60)
    
    model = load_embedding_model("all-MiniLM-L6-v2")
    embeddings = generate_embeddings(model, all_chunks)
    
    print(f"\n✅ Final Summary:")
    print(f"   Total chunks    : {len(all_chunks)}")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Vector size     : {len(embeddings[0])}")
    print(f"\n Ready for Step 4 → Store into ChromaDB!")