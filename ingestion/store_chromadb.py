# ingestion/store_chromadb.py

import chromadb
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data import load_dataset
from chunking import row_level_chunks, crop_aggregate_chunks
from embeddings import load_embedding_model, generate_embeddings

# ─────────────────────────────────────────
# Initialize ChromaDB
# ─────────────────────────────────────────

def init_chromadb(persist_directory: str = "chromadb_store"):
    """
    Initialize ChromaDB with persistent storage.
    Data stays saved even after you close the program.
    """
    
    print(f"\n Initializing ChromaDB...")
    print(f" Storage location: {persist_directory}/")
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    print(f"✅ ChromaDB initialized")
    return client


# ─────────────────────────────────────────
# Create Collection
# ─────────────────────────────────────────

def create_collection(client, collection_name: str = "crop_rag"):
    """
    Create a ChromaDB collection.
    If collection already exists, delete and recreate it fresh.
    """
    
    print(f"\n Setting up collection: '{collection_name}'")
    
    # Delete if already exists (fresh start)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f" Deleted existing collection — starting fresh")
    
    collection = client.create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"}  # cosine similarity for embeddings
    )
    
    print(f"✅ Collection '{collection_name}' created")
    return collection


# ─────────────────────────────────────────
# Store Chunks into ChromaDB
# ─────────────────────────────────────────

def store_chunks(collection, chunks: list, embeddings: list, batch_size: int = 100):
    """
    Store chunks, embeddings, and metadata into ChromaDB.
    Uses batching to avoid memory issues.
    """
    
    total = len(chunks)
    print(f"\n Storing {total} chunks into ChromaDB...")
    print(f" Batch size: {batch_size}")
    
    # Process in batches
    for i in range(0, total, batch_size):
        
        batch_chunks     = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        
        # Prepare data for ChromaDB
        ids        = []
        documents  = []
        metadatas  = []
        embeds     = []
        
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            
            # Unique ID for each chunk
            chunk_id = f"chunk_{i + j}"
            
            # Clean metadata — ChromaDB only accepts str, int, float, bool
            clean_metadata = {}
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(clean_metadata)
            embeds.append(embedding)
        
        # Add batch to ChromaDB
        collection.add(
            ids        = ids,
            documents  = documents,
            metadatas  = metadatas,
            embeddings = embeds
        )
        
        print(f" Stored batch {i // batch_size + 1} "
              f"({min(i + batch_size, total)}/{total} chunks)")
    
    print(f"✅ All {total} chunks stored successfully!")


# ─────────────────────────────────────────
# Verify Storage
# ─────────────────────────────────────────

def verify_storage(collection):
    """
    Verify data was stored correctly.
    Run a quick test query to make sure retrieval works.
    """
    
    print(f"\n" + "=" * 60)
    print(" VERIFYING STORAGE")
    print("=" * 60)
    
    # Total count
    total = collection.count()
    print(f"\n Total documents in ChromaDB : {total}")
    
    # Peek at first 2 entries
    sample = collection.peek(limit=2)
    print(f"\n Sample stored documents:")
    for i, doc in enumerate(sample["documents"]):
        print(f"\n  [{i+1}] {doc}")
        print(f"       Metadata: {sample['metadatas'][i]}")
    
    print(f"\n✅ Storage verified successfully!")


# ─────────────────────────────────────────
# Quick Test Query
# ─────────────────────────────────────────

def test_query(collection, model):
    """
    Run a quick test query to confirm retrieval works.
    """
    
    print(f"\n" + "=" * 60)
    print(" TEST QUERY")
    print("=" * 60)
    
    test_question = "What crop grows best with high nitrogen and acidic soil?"
    print(f"\n Query: {test_question}")
    
    # Embed the query
    query_embedding = model.encode(test_question).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = 3
    )
    
    print(f"\n Top 3 retrieved chunks:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n  Result {i+1}:")
        print(f"  Text     : {doc}")
        print(f"  Crop     : {meta.get('crop', 'N/A')}")
        print(f"  Distance : {round(results['distances'][0][i], 4)}")
    
    print(f"\n✅ Test query successful — retrieval is working!")


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print(" CROP RAG — INGESTION PIPELINE")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n📂 STEP 1: Loading dataset...")
    df = load_dataset("data/Crop_recommendation.csv")
    print(f"✅ Loaded {len(df)} rows")
    
    # 2. Create chunks (hybrid strategy)
    print("\n✂️  STEP 2: Creating chunks...")
    row_chunks = row_level_chunks(df)
    agg_chunks = crop_aggregate_chunks(df)
    all_chunks = row_chunks + agg_chunks
    print(f"✅ Row-level: {len(row_chunks)} | "
          f"Aggregate: {len(agg_chunks)} | "
          f"Total: {len(all_chunks)}")
    
    # 3. Generate embeddings
    print("\n🔢 STEP 3: Generating embeddings...")
    model      = load_embedding_model("all-MiniLM-L6-v2")
    embeddings = generate_embeddings(model, all_chunks)
    print(f"✅ {len(embeddings)} embeddings generated")
    
    # 4. Store into ChromaDB
    print("\n💾 STEP 4: Storing into ChromaDB...")
    client     = init_chromadb("chromadb_store")
    collection = create_collection(client, "crop_rag")
    store_chunks(collection, all_chunks, embeddings)
    
    # 5. Verify
    verify_storage(collection)
    
    # 6. Test query
    test_query(collection, model)
    
    print("\n" + "=" * 60)
    print("🎉 INGESTION PIPELINE COMPLETE!")
    print(" ChromaDB is ready at: chromadb_store/")
    print(" Next → Step 5: Retrieval Pipeline")
    print("=" * 60)