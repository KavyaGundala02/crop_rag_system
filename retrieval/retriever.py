# retrieval/retriever.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# ─────────────────────────────────────────
# Load ChromaDB Collection
# ─────────────────────────────────────────

def load_collection(
    persist_directory : str = "chromadb_store",
    collection_name   : str = "crop_rag"
):
    """Load existing ChromaDB collection."""
    
    print(f"\n Loading ChromaDB collection: '{collection_name}'")
    
    client     = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)
    
    print(f"✅ Collection loaded — {collection.count()} documents")
    return collection


# ─────────────────────────────────────────
# Method 1 — Semantic Search
# ─────────────────────────────────────────

def semantic_search(
    collection,
    model,
    query   : str,
    top_k   : int = 5
) -> list:
    """
    Pure semantic search using embedding similarity.
    Converts query to vector → finds closest chunks in ChromaDB.
    """
    
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k
    )
    
    # Format results
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text"     : results["documents"][0][i],
            "metadata" : results["metadatas"][0][i],
            "score"    : round(1 - results["distances"][0][i], 4),
            "method"   : "semantic"
        })
    
    return retrieved


# ─────────────────────────────────────────
# Method 2 — Semantic + Metadata Filter
# ─────────────────────────────────────────

def filtered_search(
    collection,
    model,
    query   : str,
    filters : dict,
    top_k   : int = 5
) -> list:
    """
    Semantic search with metadata filtering.
    Example filters:
        {"ph": {"$lt": 6.0}}           → only crops with pH < 6
        {"crop": {"$eq": "rice"}}       → only rice
        {"rainfall": {"$gt": 200.0}}    → only high rainfall crops
    """
    
    print(f"\n Applying metadata filter: {filters}")
    
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search with filter
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        where            = filters
    )
    
    # Format results
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text"     : results["documents"][0][i],
            "metadata" : results["metadatas"][0][i],
            "score"    : round(1 - results["distances"][0][i], 4),
            "method"   : "filtered_semantic"
        })
    
    return retrieved


# ─────────────────────────────────────────
# Method 3 — Hybrid BM25 + Semantic
# ─────────────────────────────────────────

def build_bm25_index(collection):
    """
    Build BM25 index from all documents in ChromaDB.
    BM25 is keyword-based search (complements semantic).
    """
    
    print(f"\n Building BM25 index...")
    
    # Get all documents from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])
    
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    
    # Tokenize for BM25
    tokenized = [doc.lower().split() for doc in documents]
    bm25      = BM25Okapi(tokenized)
    
    print(f"✅ BM25 index built with {len(documents)} documents")
    return bm25, documents, metadatas


def hybrid_search(
    collection,
    model,
    bm25,
    all_documents : list,
    all_metadatas : list,
    query         : str,
    top_k         : int = 5,
    bm25_weight   : float = 0.2,
    semantic_weight: float = 0.8
) -> list:
    """
    Hybrid search: BM25 + Semantic with Reciprocal Rank Fusion.
    Combines keyword matching and semantic similarity.
    
    bm25_weight + semantic_weight should = 1.0
    Default: 30% BM25, 70% semantic
    """
    
    total_docs = len(all_documents)
    
    # ── BM25 scores ──
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores to 0-1
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # ── Semantic scores ──
    query_embedding  = model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = total_docs  # get all docs
    )
    
    # Map semantic scores back to doc index
    semantic_scores = np.zeros(total_docs)
    all_ids         = collection.get()["ids"]
    
    for doc_id, distance in zip(
        semantic_results["ids"][0],
        semantic_results["distances"][0]
    ):
        idx = all_ids.index(doc_id)
        semantic_scores[idx] = 1 - distance  # convert distance to similarity
    
    # ── Combine scores ──
    combined_scores = (
        bm25_weight     * bm25_scores +
        semantic_weight * semantic_scores
    )
    
    # Get top_k indices
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    # Format results
    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "text"          : all_documents[idx],
            "metadata"      : all_metadatas[idx],
            "score"         : round(float(combined_scores[idx]), 4),
            "bm25_score"    : round(float(bm25_scores[idx]), 4),
            "semantic_score": round(float(semantic_scores[idx]), 4),
            "method"        : "hybrid"
        })
    
    return retrieved


# ─────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────

def display_results(results: list, query: str):
    """Pretty print retrieval results."""
    
    print(f"\n" + "=" * 60)
    print(f" Query  : {query}")
    print(f" Method : {results[0]['method']}")
    print(f" Top-{len(results)} Results:")
    print("=" * 60)
    
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] Score : {r['score']}")
        print(f"       Crop  : {r['metadata'].get('crop', 'N/A')}")
        print(f"       Text  : {r['text'][:120]}...")
        
        # Show extra scores for hybrid
        if r["method"] == "hybrid":
            print(f"       BM25  : {r['bm25_score']} | "
                  f"Semantic: {r['semantic_score']}")


# ─────────────────────────────────────────
# Compare Retrieval Methods
# ─────────────────────────────────────────

def compare_retrieval(collection, model, query: str, top_k: int = 5):
    """
    Run all 3 retrieval methods on same query.
    Compare which crops are retrieved by each method.
    """
    
    print(f"\n" + "=" * 60)
    print(f" COMPARING RETRIEVAL METHODS")
    print(f" Query : {query}")
    print("=" * 60)
    
    # Method 1 — Semantic
    print(f"\n🔹 METHOD 1: SEMANTIC SEARCH")
    semantic_results = semantic_search(collection, model, query, top_k)
    crops_semantic   = [r["metadata"].get("crop") for r in semantic_results]
    print(f" Crops retrieved: {crops_semantic}")
    
    # Method 2 — Hybrid
    print(f"\n🔹 METHOD 2: HYBRID SEARCH (BM25 + Semantic)")
    bm25, all_docs, all_metas = build_bm25_index(collection)
    hybrid_results  = hybrid_search(
        collection, model, bm25,
        all_docs, all_metas, query, top_k
    )
    crops_hybrid = [r["metadata"].get("crop") for r in hybrid_results]
    print(f" Crops retrieved: {crops_hybrid}")
    
    # Method 3 — Filtered
    print(f"\n🔹 METHOD 3: FILTERED SEARCH (pH < 6.5)")
    filtered_results = filtered_search(
        collection, model, query,
        filters = {"ph": {"$lt": 6.5}},
        top_k   = top_k
    )
    crops_filtered = [r["metadata"].get("crop") for r in filtered_results]
    print(f" Crops retrieved: {crops_filtered}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f" COMPARISON SUMMARY")
    print("=" * 60)
    print(f" Semantic only : {crops_semantic}")
    print(f" Hybrid        : {crops_hybrid}")
    print(f" Filtered      : {crops_filtered}")
    
    return {
        "semantic" : semantic_results,
        "hybrid"   : hybrid_results,
        "filtered" : filtered_results
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print(" CROP RAG — RETRIEVAL PIPELINE")
    print("=" * 60)
    
    # Load collection and model
    collection = load_collection("chromadb_store", "crop_rag")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test queries
    queries = [
        "What crop grows best with high nitrogen and acidic soil?",
        "Which crops need low rainfall and high temperature?",
        "Compare nitrogen requirements for rice vs wheat"
    ]
    
    # Run comparison on first query
    results = compare_retrieval(
        collection, model,
        query = queries[0],
        top_k = 5
    )
    
    # Show detailed results for semantic
    print(f"\n📋 DETAILED SEMANTIC RESULTS:")
    display_results(results["semantic"], queries[0])
    
    print(f"\n📋 DETAILED HYBRID RESULTS:")
    display_results(results["hybrid"], queries[0])
    
    print("\n" + "=" * 60)
    print("🎉 RETRIEVAL PIPELINE COMPLETE!")
    print(" Next → Step 6: Generation with Gemini")
    print("=" * 60)