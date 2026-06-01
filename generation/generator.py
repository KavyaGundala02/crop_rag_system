# generation/generator.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieval'))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time
from retriever import (
    load_collection,
    semantic_search,
    hybrid_search,
    build_bm25_index
)

load_dotenv()

# ─────────────────────────────────────────
# Initialize LLM with Fallback
# ─────────────────────────────────────────

def init_llm() -> dict:
    """
    Try Gemini first.
    If Gemini fails or key missing → fall back to Groq.
    
    Returns dict:
    {
        "client"  : <client object>,
        "provider": "gemini" or "groq",
        "model"   : <model name>
    }
    """
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")
    
    print("\n" + "=" * 60)
    print(" LLM INITIALIZATION")
    print("=" * 60)
    
    # ── Try Gemini First ──
    if gemini_key:
        print(f"\n🔄 GEMINI_API_KEY found — testing connection...")
        
        try:
            from google import genai
            
            client   = genai.Client(api_key=gemini_key)
            
            # Quick test call
            response = client.models.generate_content(
                model    = "models/gemini-2.0-flash-lite",
                contents = "say ok in one word"
            )
            
            print(f"✅ Gemini connected!")
            print(f"   Provider : Gemini")
            print(f"   Model    : gemini-2.0-flash-lite")
            print("=" * 60)
            
            return {
                "client"  : client,
                "provider": "gemini",
                "model"   : "models/gemini-2.0-flash-lite"
            }
        
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"⚠️  Gemini quota exceeded!")
            elif "401" in error_msg or "UNAUTHENTICATED" in error_msg:
                print(f"⚠️  Gemini API key is invalid!")
            elif "404" in error_msg or "NOT_FOUND" in error_msg:
                print(f"⚠️  Gemini model not found!")
            else:
                print(f"⚠️  Gemini error: {error_msg[:80]}")
            
            print("🔄 Falling back to Groq...")
    
    else:
        print("⚠️  No GEMINI_API_KEY found in .env")
        print("🔄 Trying Groq...")
    
    # ── Fall Back to Groq ──
    if groq_key:
        print(f"\n🔄 GROQ_API_KEY found — testing connection...")
        
        try:
            from groq import Groq
            
            client   = Groq(api_key=groq_key)
            
            # Quick test call
            response = client.chat.completions.create(
                model      = "llama-3.1-8b-instant",
                messages   = [{"role": "user", "content": "say ok"}],
                max_tokens = 5
            )
            
            print(f"✅ Groq connected!")
            print(f"   Provider : Groq")
            print(f"   Model    : llama-3.1-8b-instant")
            print("=" * 60)
            
            return {
                "client"  : client,
                "provider": "groq",
                "model"   : "llama-3.1-8b-instant"
            }
        
        except Exception as e:
            raise RuntimeError(
                f"❌ Groq also failed: {str(e)}\n"
                f"Please check your GROQ_API_KEY in .env"
            )
    
    else:
        raise RuntimeError(
            "❌ No API keys found!\n"
            "Please add GEMINI_API_KEY or GROQ_API_KEY to your .env file"
        )


# ─────────────────────────────────────────
# Build Prompt
# ─────────────────────────────────────────

def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Build grounded prompt using retrieved chunks.
    LLM will ONLY answer from these chunks.
    """
    
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        crop  = chunk["metadata"].get("crop", "unknown")
        score = chunk["score"]
        context_text += (
            f"\n[Source {i+1}] "
            f"(Crop: {crop}, Relevance: {score})\n"
            f"{chunk['text']}\n"
        )
    
    prompt = f"""You are an expert agricultural assistant.
Answer the user's question using ONLY the data sources provided below.

STRICT RULES:
1. Only use information from the provided sources
2. Always cite which Source number supports your answer
3. If the answer is not in the sources say "I don't have enough data"
4. Never make up or assume agronomic facts
5. Be specific with numbers (N, P, K, pH, rainfall values)

─────────────────────────────────
DATA SOURCES:
{context_text}
─────────────────────────────────

USER QUESTION: {query}

ANSWER (cite sources):"""
    
    return prompt


# ─────────────────────────────────────────
# Generate with Gemini
# ─────────────────────────────────────────

def _generate_with_gemini(client, model: str, prompt: str) -> str:
    """Generate answer using Gemini client."""
    
    response = client.models.generate_content(
        model    = model,
        contents = prompt
    )
    return response.text.strip()


# ─────────────────────────────────────────
# Generate with Groq
# ─────────────────────────────────────────

def _generate_with_groq(client, model: str, prompt: str) -> str:
    """Generate answer using Groq client."""
    
    response = client.chat.completions.create(
        model    = model,
        messages = [
            {
                "role"   : "system",
                "content": "You are an expert agricultural assistant. Answer strictly from provided data sources only."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature = 0.1,
        max_tokens  = 1024
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# Generate Answer (handles both providers)
# ─────────────────────────────────────────

def generate_answer(
    llm             : dict,
    query           : str,
    retrieved_chunks: list
) -> dict:
    """
    Generate grounded answer.
    Automatically uses Gemini or Groq based on llm dict.
    If Gemini fails mid-session → auto switches to Groq.
    """
    
    prompt   = build_prompt(query, retrieved_chunks)
    provider = llm["provider"]
    model    = llm["model"]
    answer   = None
    
    # ── Gemini ──
    if provider == "gemini":
        try:
            answer = _generate_with_gemini(
                llm["client"], model, prompt
            )
            used_provider = "gemini"
        
        except Exception as e:
            error_msg = str(e)
            
            # Gemini failed mid-session → switch to Groq
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"\n⚠️  Gemini quota hit mid-session!")
            elif "401" in error_msg or "UNAUTHENTICATED" in error_msg:
                print(f"\n⚠️  Gemini auth failed mid-session!")
            else:
                print(f"\n⚠️  Gemini failed: {error_msg[:80]}")
            
            print("🔄 Auto switching to Groq for this query...")
            
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise RuntimeError(
                    "❌ Gemini failed and no GROQ_API_KEY fallback in .env"
                )
            
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
            answer      = _generate_with_groq(
                groq_client,
                "llama-3.1-8b-instant",
                prompt
            )
            used_provider = "groq (auto fallback)"
    
    # ── Groq ──
    elif provider == "groq":
        answer        = _generate_with_groq(
            llm["client"], model, prompt
        )
        used_provider = "groq"
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return {
        "query"    : query,
        "answer"   : answer,
        "provider" : used_provider,
        "model"    : model,
        "sources"  : retrieved_chunks,
        "prompt"   : prompt
    }


# ─────────────────────────────────────────
# Compare Prompts
# ─────────────────────────────────────────

def compare_prompts(
    llm   : dict,
    query : str,
    chunks: list
):
    """Compare 2 prompt strategies."""
    
    print(f"\n{'='*60}")
    print(f" PROMPT COMPARISON")
    print(f"{'='*60}")
    
    context   = "\n".join([c["text"] for c in chunks])
    
    # ── Prompt V1 — Basic ──
    prompt_v1 = f"""Based on this crop data:
{context}
Answer this question: {query}"""
    
    if llm["provider"] == "gemini":
        resp_v1 = _generate_with_gemini(
            llm["client"], llm["model"], prompt_v1
        )
    else:
        resp_v1 = _generate_with_groq(
            llm["client"], llm["model"], prompt_v1
        )
    
    print(f"\n🔹 PROMPT V1 (Basic):")
    print(f" {resp_v1[:300]}...")
    
    # ── Prompt V2 — Structured ──
    prompt_v2 = build_prompt(query, chunks)
    
    if llm["provider"] == "gemini":
        resp_v2 = _generate_with_gemini(
            llm["client"], llm["model"], prompt_v2
        )
    else:
        resp_v2 = _generate_with_groq(
            llm["client"], llm["model"], prompt_v2
        )
    
    print(f"\n🔹 PROMPT V2 (Structured + Citations):")
    print(f" {resp_v2[:300]}...")
    
    print(f"\n📊 VERDICT:")
    print(f" V1 — Simple, may hallucinate, no citations")
    print(f" V2 — Structured, grounded, cites sources ✅")


# ─────────────────────────────────────────
# Full RAG Pipeline
# ─────────────────────────────────────────

def rag_query(
    query      : str,
    collection ,
    embed_model,
    llm        : dict,
    bm25             = None,
    all_docs   : list = None,
    all_metas  : list = None,
    top_k      : int  = 5,
    use_hybrid : bool = True
) -> dict:
    """Full RAG pipeline: Query → Retrieve → Generate"""
    
    print(f"\n{'='*60}")
    print(f" RAG QUERY")
    print(f" Q: {query}")
    print(f"{'='*60}")
    
    # ── Retrieve ──
    print(f"\n🔍 Retrieving top-{top_k} chunks...")
    
    if use_hybrid and bm25 is not None:
        chunks = hybrid_search(
            collection, embed_model,
            bm25, all_docs, all_metas,
            query, top_k
        )
        print(f" Method: Hybrid (BM25 + Semantic)")
    else:
        chunks = semantic_search(
            collection, embed_model,
            query, top_k
        )
        print(f" Method: Semantic only")
    
    crops = [c["metadata"].get("crop") for c in chunks]
    print(f" Crops retrieved: {crops}")
    
    # ── Generate ──
    print(f"\n🤖 Generating with {llm['provider'].upper()}...")
    result = generate_answer(llm, query, chunks)
    
    print(f"\n📋 ANSWER ({result['provider'].upper()}):")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    
    return result


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print(" CROP RAG — GENERATION PIPELINE")
    print("=" * 60)
    
    # Load everything
    collection  = load_collection("chromadb_store", "crop_rag")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # ── Auto detect LLM ──
    llm = init_llm()
    print(f"\n🟢 Active LLM : {llm['provider'].upper()}")
    print(f"   Model      : {llm['model']}")
    
    # Build BM25
    bm25, all_docs, all_metas = build_bm25_index(collection)
    
    # Test queries
    test_queries = [
        "What crop grows best with high nitrogen and acidic soil?",
        "Which crops are suitable for humid tropical climate?",
        "Compare nitrogen requirements for rice vs wheat",
        "What should I grow if my soil pH is 5.5 and rainfall is 250mm?"
    ]
    
    # Run RAG
    for query in test_queries:
        result = rag_query(
            query      = query,
            collection = collection,
            embed_model= embed_model,
            llm        = llm,
            bm25       = bm25,
            all_docs   = all_docs,
            all_metas  = all_metas,
            top_k      = 5,
            use_hybrid = True
        )
    
    # Compare prompts
    print("\n🧪 RUNNING PROMPT COMPARISON...")
    sample_chunks = semantic_search(
        collection, embed_model,
        test_queries[0], top_k=5
    )
    compare_prompts(llm, test_queries[0], sample_chunks)
    
    print("\n" + "=" * 60) 
    print("🎉 GENERATION PIPELINE COMPLETE!")
    print("=" * 60)