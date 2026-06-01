# evaluation/evaluator.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieval'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generation'))

import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from retriever import (
    load_collection,
    semantic_search,
    hybrid_search,
    build_bm25_index
)
from generator import init_llm, generate_answer, build_prompt

load_dotenv()

# ─────────────────────────────────────────
# Test Set (25 Q&A pairs)
# ─────────────────────────────────────────

# Valid 22 crops in dataset:
# rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans,
# mungbean, blackgram, lentil, pomegranate, banana, mango,
# grapes, watermelon, muskmelon, apple, orange, papaya,
# coconut, cotton, jute, coffee

TEST_SET = [
    {
        "question"      : "What crop grows best with high nitrogen and acidic soil?",
        "answer"        : "Coffee grows best with high nitrogen and acidic soil.",
        "expected_crops": ["coffee"]
    },
    {
        "question"      : "Which crop needs the highest potassium levels?",
        "answer"        : "Grapes require very high potassium levels.",
        "expected_crops": ["grapes"]
    },
    {
        "question"      : "What should I grow with pH 5.5 and rainfall 250mm?",
        "answer"        : "Rice or maize are suitable for pH 5.5 and 250mm rainfall.",
        "expected_crops": ["rice", "maize"]
    },
    {
        "question"      : "Which crops are suitable for humid tropical climate?",
        "answer"        : "Rice, coconut, banana and papaya suit humid tropical climates.",
        "expected_crops": ["rice", "coconut", "banana", "papaya"]
    },
    {
        "question"      : "Compare nitrogen requirements for rice vs maize",
        # FIXED: wheat → maize (wheat not in dataset)
        "answer"        : "Rice needs slightly more nitrogen than maize on average.",
        "expected_crops": ["rice", "maize"]
    },
    {
        "question"      : "Which crop needs the least rainfall?",
        "answer"        : "Chickpea and mothbeans need the least rainfall.",
        "expected_crops": ["chickpea", "mothbeans"]
    },
    {
        "question"      : "What crop grows in very high humidity?",
        "answer"        : "Coconut and rice grow in very high humidity conditions.",
        "expected_crops": ["coconut", "rice"]
    },
    {
        "question"      : "Which crop tolerates the highest temperature?",
        "answer"        : "Papaya and mango can tolerate the highest temperatures.",
        "expected_crops": ["papaya", "mango"]
    },
    {
        "question"      : "What crops grow in low nitrogen soil?",
        "answer"        : "Mungbean and lentil grow well in low nitrogen soil.",
        "expected_crops": ["mungbean", "lentil"]
    },
    {
        "question"      : "Which crop needs the highest phosphorus?",
        "answer"        : "Grapes and banana need high phosphorus levels.",
        "expected_crops": ["grapes", "banana"]
    },
    {
        "question"      : "What crop is best for alkaline soil?",
        "answer"        : "Cotton grows well in alkaline soil conditions.",
        "expected_crops": ["cotton"]
    },
    {
        "question"      : "Which crops grow in low temperature?",
        "answer"        : "Apple and grapes can grow in lower temperatures.",
        "expected_crops": ["apple", "grapes"]
    },
    {
        "question"      : "What crop needs high rainfall above 200mm?",
        "answer"        : "Rice and coconut need high rainfall above 200mm.",
        "expected_crops": ["rice", "coconut"]
    },
    {
        "question"      : "Which crops need balanced NPK nutrients?",
        # FIXED: wheat → jute (wheat not in dataset)
        "answer"        : "Maize and jute need balanced NPK nutrients.",
        "expected_crops": ["maize", "jute"]
    },
    {
        "question"      : "What crop grows with very low phosphorus?",
        "answer"        : "Jute and cotton grow with low phosphorus levels.",
        "expected_crops": ["jute", "cotton"]
    },
    {
        "question"      : "Which crop is best for sandy soil with low pH?",
        "answer"        : "Coffee is suitable for sandy acidic soil.",
        "expected_crops": ["coffee"]
    },
    {
        "question"      : "What is the ideal pH for banana cultivation?",
        "answer"        : "Banana grows best in slightly acidic to neutral pH around 6.",
        "expected_crops": ["banana"]
    },
    {
        "question"      : "Which crops grow in dry weather?",
        "answer"        : "Chickpea, mothbeans and pigeonpeas grow in dry weather.",
        "expected_crops": ["chickpea", "mothbeans", "pigeonpeas"]
    },
    {
        "question"      : "What crop needs moderate temperature around 25 degrees?",
        "answer"        : "Maize and cotton grow best at moderate temperatures.",
        "expected_crops": ["maize", "cotton"]
    },
    {
        "question"      : "Which crop has highest humidity requirement?",
        "answer"        : "Coconut has the highest humidity requirement.",
        "expected_crops": ["coconut"]
    },
    {
        "question"      : "What crop suits low potassium soil?",
        "answer"        : "Chickpea and lentil suit low potassium soil.",
        "expected_crops": ["chickpea", "lentil"]
    },
    {
        "question"      : "Which fruits grow in tropical climate?",
        "answer"        : "Mango, banana, papaya, coconut grow in tropical climate.",
        "expected_crops": ["mango", "banana", "papaya", "coconut"]
    },
    {
        "question"      : "What is the nitrogen range for cotton?",
        "answer"        : "Cotton requires moderate nitrogen levels around 100-120.",
        "expected_crops": ["cotton"]
    },
    {
        "question"      : "Which crop needs the most water?",
        "answer"        : "Rice needs the most water among all crops.",
        "expected_crops": ["rice"]
    },
    {
        "question"      : "What crop grows with high potassium and low nitrogen?",
        "answer"        : "Grapes grow with high potassium and relatively low nitrogen.",
        "expected_crops": ["grapes"]
    }
]


# ─────────────────────────────────────────
# Retrieval Metrics
# ─────────────────────────────────────────

# In evaluator.py replace all metric functions

def calculate_hit_rate(retrieved: list, expected: list) -> float:
    retrieved_set = set(r.lower().strip() for r in retrieved)
    expected_set  = set(e.lower().strip() for e in expected)
    return 1.0 if retrieved_set & expected_set else 0.0


def calculate_precision_at_k(retrieved: list, expected: list) -> float:
    if not retrieved:
        return 0.0
    expected_set = set(e.lower().strip() for e in expected)
    hits = sum(1 for c in retrieved if c.lower().strip() in expected_set)
    return round(hits / len(retrieved), 4)


def calculate_recall_at_k(retrieved: list, expected: list) -> float:
    if not expected:
        return 0.0
    expected_set = set(e.lower().strip() for e in expected)
    hits = sum(1 for c in retrieved if c.lower().strip() in expected_set)
    return round(hits / len(expected_set), 4)


def calculate_mrr(retrieved: list, expected: list) -> float:
    expected_set = set(e.lower().strip() for e in expected)
    for rank, crop in enumerate(retrieved, start=1):
        if crop.lower().strip() in expected_set:
            return round(1.0 / rank, 4)
    return 0.0

# ─────────────────────────────────────────
# LLM as Judge (works with both Gemini/Groq)
# ─────────────────────────────────────────

def evaluate_answer_with_llm(
    llm      : dict,
    question : str,
    generated: str,
    expected : str,
    context  : str
) -> dict:
    """
    Use active LLM (Gemini or Groq) as judge.
    Scores: Faithfulness, Relevance, Hallucination, Correctness
    """
    
    judge_prompt = f"""You are an evaluation judge for an agricultural RAG system.
Score the generated answer on 4 metrics.
Return ONLY a JSON object, nothing else. No explanation. No markdown.

QUESTION: {question}
EXPECTED ANSWER: {expected}
GENERATED ANSWER: {generated}
CONTEXT USED: {context[:400]}

Return ONLY this JSON:
{{
  "faithfulness"  : <0.0-1.0, is answer grounded in context?>,
  "relevance"     : <0.0-1.0, does answer address the question?>,
  "hallucination" : <0.0-1.0, 0=no hallucination 1=full hallucination>,
  "correctness"   : <0.0-1.0, how correct vs expected answer?>
}}"""
    
    # ── Use Gemini ──
    if llm["provider"] == "gemini":
        try:
            response = llm["client"].models.generate_content(
                model    = llm["model"],
                contents = judge_prompt
            )
            raw = response.text.strip()
        except Exception as e:
            print(f"   ⚠️  Gemini judge failed: {str(e)[:60]}")
            print(f"   🔄 Using Groq as judge fallback...")
            raw = _groq_judge(judge_prompt)
    
    # ── Use Groq ──
    else:
        raw = _groq_judge_with_client(llm["client"], judge_prompt)
    
    # Parse JSON
    try:
        raw    = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
    except Exception as e:
        print(f"   ⚠️  Could not parse judge response: {e}")
        scores = {
            "faithfulness" : 0.5,
            "relevance"    : 0.5,
            "hallucination": 0.5,
            "correctness"  : 0.5
        }
    
    return scores


def _groq_judge(prompt: str) -> str:
    """Groq fallback for judge when Gemini fails."""
    from groq import Groq
    groq_key = os.getenv("GROQ_API_KEY")
    client   = Groq(api_key=groq_key)
    return _groq_judge_with_client(client, prompt)


def _groq_judge_with_client(client, prompt: str) -> str:
    """Use existing Groq client for judge."""
    response = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        messages   = [{"role": "user", "content": prompt}],
        temperature= 0.0,
        max_tokens = 200
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# Run Full Evaluation
# ─────────────────────────────────────────

def run_evaluation(
    collection,
    embed_model,
    llm       : dict,
    bm25,
    all_docs  : list,
    all_metas : list,
    top_k     : int  = 5,
    use_hybrid: bool = True
) -> list:
    """Run full evaluation on all 25 test questions."""
    
    print("\n" + "=" * 60)
    print(f" RUNNING EVALUATION — {len(TEST_SET)} QUESTIONS")
    print(f" LLM      : {llm['provider'].upper()}")
    print(f" Method   : {'Hybrid' if use_hybrid else 'Semantic'}")
    print(f" Top-K    : {top_k}")
    print("=" * 60)
    
    all_results = []
    
    for i, test in enumerate(TEST_SET):
        
        question        = test["question"]
        expected_answer = test["answer"]
        expected_crops  = test["expected_crops"]
        
        print(f"\n[{i+1}/{len(TEST_SET)}] {question[:55]}...")
        
        # ── Retrieve ──
        if use_hybrid:
            chunks = hybrid_search(
                collection, embed_model,
                bm25, all_docs, all_metas,
                question, top_k
            )
        else:
            chunks = semantic_search(
                collection, embed_model,
                question, top_k
            )
        
        retrieved_crops = [
            c["metadata"].get("crop", "") for c in chunks
        ]
        
        # ── Retrieval Metrics ──
        hit_rate  = calculate_hit_rate(retrieved_crops, expected_crops)
        precision = calculate_precision_at_k(retrieved_crops, expected_crops)
        recall    = calculate_recall_at_k(retrieved_crops, expected_crops)
        mrr       = calculate_mrr(retrieved_crops, expected_crops)
        
        # ── Generate ──
        result    = generate_answer(llm, question, chunks)
        generated = result["answer"]
        context   = " ".join([c["text"] for c in chunks[:3]])
        
        # ── Generation Metrics ──
        gen_scores = evaluate_answer_with_llm(
            llm, question,
            generated, expected_answer, context
        )
        
        row = {
            "question"        : question,
            "expected_crops"  : str(expected_crops),
            "retrieved_crops" : str(retrieved_crops),
            "hit_rate"        : hit_rate,
            "precision_k"     : precision,
            "recall_k"        : recall,
            "mrr"             : mrr,
            "faithfulness"    : gen_scores.get("faithfulness",  0.0),
            "relevance"       : gen_scores.get("relevance",     0.0),
            "hallucination"   : gen_scores.get("hallucination", 1.0),
            "correctness"     : gen_scores.get("correctness",   0.0),
            "llm_provider"    : result.get("provider", llm["provider"]),
            "generated_answer": generated
        }
        
        all_results.append(row)
        
        print(
            f"   Hit:{hit_rate} | "
            f"P@K:{precision} | "
            f"R@K:{recall} | "
            f"MRR:{mrr} | "
            f"Faith:{gen_scores.get('faithfulness', 0.0)} | "
            f"LLM:{result.get('provider', llm['provider'])}"
        )
    
    return all_results


# ─────────────────────────────────────────
# Generate Report
# ─────────────────────────────────────────

def generate_report(all_results: list, label: str = "Experiment"):
    """Summarise metrics into a report table."""
    
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*60}")
    print(f" EVALUATION REPORT — {label}")
    print(f"{'='*60}")
    
    print(f"\n📊 RETRIEVAL METRICS (avg over {len(df)} questions):")
    print(f"   Hit Rate   : {round(df['hit_rate'].mean(),   4)}")
    print(f"   Precision@K: {round(df['precision_k'].mean(),4)}")
    print(f"   Recall@K   : {round(df['recall_k'].mean(),   4)}")
    print(f"   MRR        : {round(df['mrr'].mean(),        4)}")
    
    print(f"\n📊 GENERATION METRICS (avg over {len(df)} questions):")
    print(f"   Faithfulness  : {round(df['faithfulness'].mean(),  4)}")
    print(f"   Relevance     : {round(df['relevance'].mean(),     4)}")
    print(f"   Hallucination : {round(df['hallucination'].mean(), 4)}")
    print(f"   Correctness   : {round(df['correctness'].mean(),   4)}")
    
    # Failure cases
    failures = df[df["hit_rate"] == 0.0]
    print(f"\n❌ FAILURE CASES ({len(failures)}):")
    for _, row in failures.iterrows():
        print(f"   Q: {row['question'][:55]}...")
        print(f"   Expected : {row['expected_crops']}")
        print(f"   Got      : {row['retrieved_crops']}")
    
    # Save
    os.makedirs("evaluation", exist_ok=True)
    report_path = "evaluation/eval_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\n✅ Report saved: {report_path}")
    
    return df


# ─────────────────────────────────────────
# Compare Experiments
# ─────────────────────────────────────────

def compare_experiments(
    collection, embed_model, llm,
    bm25, all_docs, all_metas
):
    """Compare semantic vs hybrid across top_k values."""
    
    print(f"\n{'='*60}")
    print(" EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    
    experiments = [
        {"label": "Semantic | K=3",  "use_hybrid": False, "top_k": 3},
        {"label": "Semantic | K=5",  "use_hybrid": False, "top_k": 5},
        {"label": "Semantic | K=10", "use_hybrid": False, "top_k": 10},
        {"label": "Hybrid   | K=5",  "use_hybrid": True,  "top_k": 5},
        {"label": "Hybrid   | K=10", "use_hybrid": True,  "top_k": 10},
    ]
    
    summary = []
    
    for exp in experiments:
        print(f"\n🧪 Running: {exp['label']}")
        
        results = run_evaluation(
            collection, embed_model, llm,
            bm25, all_docs, all_metas,
            top_k      = exp["top_k"],
            use_hybrid = exp["use_hybrid"]
        )
        
        df = pd.DataFrame(results)
        summary.append({
            "Experiment"  : exp["label"],
            "LLM"         : llm["provider"].upper(),
            "Hit Rate"    : round(df["hit_rate"].mean(),    4),
            "Precision@K" : round(df["precision_k"].mean(), 4),
            "Recall@K"    : round(df["recall_k"].mean(),    4),
            "MRR"         : round(df["mrr"].mean(),         4),
            "Faithfulness": round(df["faithfulness"].mean(),4),
            "Correctness" : round(df["correctness"].mean(), 4),
        })
    
    summary_df = pd.DataFrame(summary)
    
    print(f"\n{'='*60}")
    print(" EXPERIMENT COMPARISON TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(
        "evaluation/experiment_comparison.csv", index=False
    )
    print(f"\n✅ Saved: evaluation/experiment_comparison.csv")
    
    return summary_df


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print(" CROP RAG — EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load everything
    collection  = load_collection("chromadb_store", "crop_rag")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Auto detect LLM
    llm = init_llm()
    print(f"\n🟢 Active LLM : {llm['provider'].upper()}")
    print(f"   Model      : {llm['model']}")
    
    # Build BM25
    bm25, all_docs, all_metas = build_bm25_index(collection)
    
    # Full evaluation
    print("\n🔬 RUNNING FULL EVALUATION...")
    results = run_evaluation(
        collection, embed_model, llm,
        bm25, all_docs, all_metas,
        top_k=10, use_hybrid=True
    )
    generate_report(results, label=f"{llm['provider'].upper()} | Hybrid | K=5")
    
    # Experiment comparison
    print("\n🧪 RUNNING EXPERIMENT COMPARISON...")
    compare_experiments(
        collection, embed_model, llm,
        bm25, all_docs, all_metas
    )
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print(f" LLM Used : {llm['provider'].upper()}")
    print(" Reports  : evaluation/")
    print(" Next     → Step 8: Streamlit UI")
    print("=" * 60)