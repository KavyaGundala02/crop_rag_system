# demo/app.py

import sys
import os

# ── Fix all paths relative to project root ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'ingestion'))
sys.path.append(os.path.join(BASE_DIR, 'retrieval'))
sys.path.append(os.path.join(BASE_DIR, 'generation'))

# ChromaDB absolute path
CHROMA_PATH = os.path.join(BASE_DIR, 'chromadb_store')

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(os.path.join(BASE_DIR, '.env'))

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────

st.set_page_config(
    page_title = "Crop Recommendation RAG",
    page_icon  = "🌾",
    layout     = "wide"
)

# ─────────────────────────────────────────
# Load All Resources (cached)
# ─────────────────────────────────────────

@st.cache_resource
def load_all_resources():
    """
    Load all models and indexes once and cache them.
    Auto detects Gemini or Groq based on .env keys.
    """
    
    from retriever import load_collection, build_bm25_index
    from generator import init_llm
    
    # Load ChromaDB using absolute path
    collection  = load_collection(CHROMA_PATH, "crop_rag")
    
    # Load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Auto detect LLM
    llm = init_llm()
    
    # Build BM25 index
    bm25, all_docs, all_metas = build_bm25_index(collection)
    
    return collection, embed_model, llm, bm25, all_docs, all_metas


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

def render_sidebar(llm_provider: str):
    """Render sidebar with settings and sample questions."""
    
    st.sidebar.title("⚙️ Settings")
    
    # Show active LLM
    if llm_provider == "gemini":
        st.sidebar.success("🟢 LLM: Gemini (Active)")
    else:
        st.sidebar.info("🔵 LLM: Groq LLaMA 3.1 (Active)")
    
    st.sidebar.markdown("---")
    
    # Retrieval method
    retrieval_method = st.sidebar.radio(
        "Retrieval Method",
        ["Hybrid (BM25 + Semantic)", "Semantic Only"],
        index = 0
    )
    
    # Top K slider
    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value = 3,
        max_value = 10,
        value     = 5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Sample Questions")
    
    samples = [
        "What crop grows best with high nitrogen and acidic soil?",
        "Which crops suit humid tropical climate?",
        "Compare nitrogen requirements for rice vs maize",
        "What should I grow if pH is 5.5 and rainfall is 250mm?",
        "Which crop needs the least rainfall?",
        "What crops grow in low temperature?",
        "Which crop needs highest potassium?",
        "What crop is best for alkaline soil?",
        "Compare humidity needs of rice and coconut",
        "Which crop grows best in high rainfall above 200mm?"
    ]
    
    selected = None
    for q in samples:
        if st.sidebar.button(q, use_container_width=True):
            selected = q
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "🛠️ Built with RAG | ChromaDB | "
        "sentence-transformers | Gemini/Groq"
    )
    
    return retrieval_method, top_k, selected


# ─────────────────────────────────────────
# Display Metadata
# ─────────────────────────────────────────

def get_meta_value(meta: dict, key: str):
    """
    Try row-level key first (N, P, K).
    If not found try aggregate key (avg_N, avg_P).
    """
    val = meta.get(key) or meta.get(f"avg_{key}")
    return round(val, 2) if isinstance(val, float) else val or "N/A"


def display_sources(chunks: list, top_k: int):
    """Display retrieved source chunks with metadata."""
    
    st.markdown(f"### 📂 Retrieved Sources (Top-{top_k})")
    
    for i, chunk in enumerate(chunks):
        
        crop     = chunk["metadata"].get("crop", "N/A").capitalize()
        score    = chunk["score"]
        strategy = chunk["metadata"].get("strategy", "row_level")
        tag      = "📊 Aggregate" if strategy == "crop_aggregate" \
                   else "📋 Row Level"
        
        with st.expander(
            f"Source {i+1} | {tag} | Crop: {crop} | Score: {score}"
        ):
            st.markdown(f"**Text:** {chunk['text']}")
            
            meta = chunk["metadata"]
            
            if strategy == "crop_aggregate":
                st.info("📊 Crop summary chunk (averaged values)")
            else:
                st.info("📋 Specific row from dataset")
            
            st.markdown("**Soil & Climate Data:**")
            
            fields = [
                ("N",           "N (kg/ha)"),
                ("P",           "P (kg/ha)"),
                ("K",           "K (kg/ha)"),
                ("temperature", "Temp (°C)"),
                ("humidity",    "Humidity (%)"),
                ("ph",          "pH"),
                ("rainfall",    "Rainfall (mm)")
            ]
            
            cols = st.columns(4)
            for j, (key, label) in enumerate(fields):
                cols[j % 4].metric(label, get_meta_value(meta, key))


# ─────────────────────────────────────────
# Display Crops Summary
# ─────────────────────────────────────────

def display_crops(chunks: list):
    """Display unique crops found in results."""
    
    st.markdown("### 🌱 Crops Found in Results")
    
    crops = list(set([
        c["metadata"].get("crop", "").capitalize()
        for c in chunks
        if c["metadata"].get("crop")
    ]))
    
    if crops:
        cols = st.columns(min(len(crops), 5))
        for i, crop in enumerate(crops):
            cols[i % 5].success(f"🌾 {crop}")


# ─────────────────────────────────────────
# Display Retrieval Info
# ─────────────────────────────────────────

def display_retrieval_info(
    retrieval_method : str,
    top_k            : int,
    chunks           : list,
    llm              : dict,
    result           : dict
):
    """Display retrieval and generation info."""
    
    with st.expander("ℹ️ Query Info"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Retrieval**")
            st.markdown(f"- Method : {retrieval_method}")
            st.markdown(f"- Top-K  : {top_k}")
            st.markdown(f"- Chunks : {len(chunks)}")
        
        with col2:
            st.markdown("**🤖 Generation**")
            st.markdown(
                f"- Provider : "
                f"{result.get('provider', llm['provider']).upper()}"
            )
            st.markdown(f"- Model    : {llm['model']}")
            
            if "fallback" in result.get("provider", ""):
                st.warning("⚠️ Gemini failed — Groq used as fallback")


# ─────────────────────────────────────────
# Main App
# ─────────────────────────────────────────

def main():
    
    # ── Header ──
    st.title("🌾 Crop Recommendation RAG System")
    st.markdown(
        "Ask any question about crop recommendations "
        "grounded strictly in the Kaggle Crop Dataset "
        "(2200 rows, 22 crops)."
    )
    st.markdown("---")
    
    # ── Load Resources ──
    st.markdown("### ⏳ Loading System...")
    
    try:
        (
            collection,
            embed_model,
            llm,
            bm25,
            all_docs,
            all_metas
        ) = load_all_resources()
        
        st.success("✅ System loaded successfully!")
    
    except Exception as e:
        st.error(f"❌ Failed to load: {str(e)}")
        st.markdown("**Troubleshooting:**")
        st.code(
            "1. Make sure chromadb_store/ exists\n"
            "   → Run: python ingestion/store_chromadb.py\n\n"
            "2. Make sure .env has API key\n"
            "   → GEMINI_API_KEY or GROQ_API_KEY\n\n"
            "3. Run from project root:\n"
            "   → streamlit run demo/app.py"
        )
        return
    
    # ── Show Active LLM ──
    if llm["provider"] == "gemini":
        st.success(f"✅ Gemini Active ({llm['model']})")
    else:
        st.info(f"🔵 Groq Active — {llm['model']}")
    
    st.markdown("---")
    
    # ── Sidebar ──
    retrieval_method, top_k, selected_sample = render_sidebar(
        llm["provider"]
    )
    
    # ── Query Input ──
    st.markdown("### 💬 Ask a Question")
    
    query = st.text_input(
        "Enter your crop question:",
        value       = selected_sample or "",
        placeholder = "e.g. What crop grows best with high nitrogen?"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", use_container_width=True)
    with col2:
        clear = st.button("🗑️ Clear")
    
    if clear:
        st.rerun()
    
    # ── Run RAG Pipeline ──
    if ask_button and query:
        
        st.markdown("---")
        
        # Step 1 — Retrieve
        with st.spinner("🔍 Retrieving relevant chunks..."):
            try:
                from retriever import semantic_search, hybrid_search
                
                use_hybrid = "Hybrid" in retrieval_method
                
                if use_hybrid:
                    chunks = hybrid_search(
                        collection, embed_model,
                        bm25, all_docs, all_metas,
                        query, top_k
                    )
                else:
                    chunks = semantic_search(
                        collection, embed_model,
                        query, top_k
                    )
                
                st.toast(
                    f"✅ Retrieved {len(chunks)} chunks", icon="🔍"
                )
                
            except Exception as e:
                st.error(f"❌ Retrieval error: {e}")
                return
        
        # Step 2 — Generate
        with st.spinner(
            f"🤖 Generating answer with "
            f"{llm['provider'].upper()}..."
        ):
            try:
                from generator import generate_answer
                
                result = generate_answer(llm, query, chunks)
                answer = result["answer"]
                
                st.toast("✅ Answer generated!", icon="🤖")
                
            except Exception as e:
                st.error(f"❌ Generation error: {e}")
                return
        
        # ── Display Answer ──
        st.markdown("### 📋 Answer")
        
        provider_used = result.get("provider", llm["provider"])
        if "fallback" in provider_used:
            st.warning(
                "⚠️ Gemini quota exceeded — "
                "answer generated by Groq (fallback)"
            )
        
        st.success(answer)
        st.markdown("---")
        
        # ── Display Sources ──
        display_sources(chunks, top_k)
        st.markdown("---")
        
        # ── Display Crops ──
        display_crops(chunks)
        st.markdown("---")
        
        # ── Display Info ──
        display_retrieval_info(
            retrieval_method, top_k,
            chunks, llm, result
        )
    
    elif ask_button and not query:
        st.warning("⚠️ Please enter a question first!")


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()