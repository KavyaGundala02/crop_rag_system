# 🌾 Crop Recommendation RAG System

A production-style **Retrieval-Augmented Generation (RAG)** system that answers natural-language queries about optimal crop recommendations, grounded strictly in the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

---

## 📌 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Chunking Strategy](#chunking-strategy)
- [Embedding Models](#embedding-models)
- [Retrieval Strategy](#retrieval-strategy)
- [Generation Layer](#generation-layer)
- [Evaluation](#evaluation)
- [Experiment Results](#experiment-results)
- [Failure Case Analysis](#failure-case-analysis)
- [Tradeoffs](#tradeoffs)
- [Demo](#demo)

---

## Overview

This system accepts natural-language agricultural queries such as:

- *"What crop grows best with high nitrogen, low rainfall, and acidic soil?"*
- *"Which crops are suitable for a humid tropical climate?"*
- *"Compare nitrogen requirements for rice vs maize."*
- *"What should I grow if my soil pH is 5.5 and rainfall is 250mm?"*

The system retrieves relevant rows from the crop dataset and uses an LLM (Gemini or Groq) to generate grounded, cited answers — **strictly from the dataset**, with no hallucination.

---

## System Architecture

```
User Query (natural language)
        │
        ▼
┌───────────────────┐
│  Embedding Model  │  ← all-MiniLM-L6-v2 (sentence-transformers)
│  Query → Vector   │
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────┐
│         Retrieval Layer            │
│                                    │
│  ┌─────────────┐ ┌──────────────┐  │
│  │  ChromaDB   │ │  BM25 Index  │  │
│  │  (Semantic) │ │  (Keyword)   │  │
│  └──────┬──────┘ └──────┬───────┘  │
│         └───────┬────────┘         │
│                 ▼                  │
│       Hybrid RRF Fusion            │
│       Top-K Chunks                 │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│         Generation Layer           │
│                                    │
│  Grounded Prompt + Context         │
│  ┌──────────────┐                  │
│  │ Gemini API   │ ← tries first    │
│  │ (fallback ↓) │                  │
│  └──────┬───────┘                  │
│         │ fails                    │
│  ┌──────▼───────┐                  │
│  │  Groq API    │ ← fallback       │
│  │ LLaMA 3.1    │                  │
│  └──────┬───────┘                  │
└─────────┼──────────────────────────┘
          │
          ▼
   Grounded Answer
   with Source Citations
```

---

## Dataset

**Source:** [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| N | Nitrogen content in soil | kg/ha | 0 – 140 |
| P | Phosphorus content in soil | kg/ha | 5 – 145 |
| K | Potassium content in soil | kg/ha | 5 – 205 |
| Temperature | Average temperature | °C | 8 – 44 |
| Humidity | Relative humidity | % | 14 – 100 |
| pH | Soil pH value | 0–14 | 3.5 – 10 |
| Rainfall | Annual rainfall | mm | 20 – 300 |
| Label | Recommended crop | — | 22 types |

**22 Crop Labels:**
`rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee`

---

## Project Structure

```
crop-rag-system/
│
├── data/
│   └── Crop_recommendation.csv       # Kaggle dataset
│
├── ingestion/
│   ├── load_data.py                  # CSV loading & exploration
│   ├── chunking.py                   # Row-level & aggregate chunking
│   ├── embeddings.py                 # Embedding generation & comparison
│   └── store_chromadb.py             # ChromaDB storage pipeline
│
├── retrieval/
│   └── retriever.py                  # Semantic, filtered & hybrid search
│
├── generation/
│   └── generator.py                  # Gemini/Groq LLM with fallback
│
├── evaluation/
│   ├── evaluator.py                  # Full evaluation pipeline
│   ├── diagnose.py                   # Failure case diagnosis
│   ├── eval_report.csv               # Per-question results
│   └── experiment_comparison.csv     # Experiment comparison table
│
├── demo/
│   └── app.py                        # Streamlit UI
│
├── chromadb_store/                   # Persistent vector DB (auto-created)
│
├── .env                              # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Git | Latest |
| VS Code | Latest |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/crop-rag-system.git
cd crop-rag-system
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Set Up API Keys

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here    # https://aistudio.google.com
GROQ_API_KEY=your_groq_api_key_here        # https://console.groq.com
```

> The system tries **Gemini first** → falls back to **Groq** automatically if Gemini fails or quota is exceeded.

### Step 5 — Download Dataset

1. Go to: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Download and place `Crop_recommendation.csv` in the `data/` folder

---

## Running the Project

### Step 1 — Run Ingestion Pipeline

```bash
python ingestion/store_chromadb.py
```

This runs the full pipeline:
- Loads CSV (2200 rows)
- Creates row-level chunks (2200) + aggregate chunks (22) = **2222 total**
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores into ChromaDB persistently

### Step 2 — Test Retrieval

```bash
python retrieval/retriever.py
```

### Step 3 — Test Generation

```bash
python generation/generator.py
```

### Step 4 — Run Evaluation

```bash
python evaluation/evaluator.py
```

### Step 5 — Launch Demo UI

```bash
# Run from project root
streamlit run demo/app.py
```

Open: http://localhost:8501

---

## Chunking Strategy

Two strategies were implemented and compared:

### Strategy 1 — Row-Level Chunking
Each CSV row becomes one natural-language sentence.

```
"Rice grows well with N=90, P=42, K=43, temperature=20.8°C,
 humidity=82.0%, pH=6.5, rainfall=202.9mm."
```

- **Total chunks:** 2200
- **Pros:** Specific, exact values, good for precise queries
- **Cons:** No summary context, repetitive

### Strategy 2 — Crop Aggregate Chunking
All rows per crop are grouped and summarised with mean/range values.

```
"Rice crop summary: Nitrogen(N) avg=79.9 range=(60.0-99.0),
 Phosphorus(P) avg=47.6 range=(35.0-60.0)..."
```

- **Total chunks:** 22 (one per crop)
- **Pros:** Good for comparison queries, compact
- **Cons:** Loses individual row specificity

### Final Choice — Hybrid Strategy ✅
Both strategies combined = **2222 chunks**

| Strategy | Chunks | Best For |
|----------|--------|----------|
| Row-level | 2200 | Specific condition queries |
| Aggregate | 22 | Comparison & summary queries |
| **Hybrid** | **2222** | **Both** ✅ |

---

## Embedding Models

Two models were compared on 100 sample chunks:

| Model | Vector Size | Speed | Accuracy |
|-------|-------------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | ~3s ✅ | Good |
| `all-mpnet-base-v2` | 768 | ~12s | Better |

**Final choice: `all-MiniLM-L6-v2`**

Reasoning: 4x faster with only marginal accuracy difference for structured agronomic data. The domain-specific nature of crop data means the larger model's advantage is minimal.

---

## Retrieval Strategy

Three retrieval methods were implemented:

### Method 1 — Semantic Search
Embedding similarity using ChromaDB cosine distance.

### Method 2 — Metadata Filtering
Semantic search with ChromaDB `where` filters.

```python
# Example: only crops with pH < 6.5
filtered_search(collection, model, query, filters={"ph": {"$lt": 6.5}})
```

### Method 3 — Hybrid Search (BM25 + Semantic) ✅
Combines keyword matching (BM25) with semantic similarity.

```
Final Score = 0.2 × BM25_score + 0.8 × Semantic_score
```

**Final choice: Hybrid Search**

BM25 catches exact agronomic terms (N, P, K values) while semantic search handles natural language variations.

---

## Generation Layer

### LLM — Auto Fallback System

```
GEMINI_API_KEY present?
        │
       YES → Test Gemini → Works? → Use Gemini ✅
        │                    │
        │                   NO → Use Groq ✅
        │
       NO → Use Groq ✅
```

| Provider | Model | Free Tier |
|----------|-------|-----------|
| Gemini | gemini-2.0-flash-lite | Limited |
| **Groq** | **llama-3.1-8b-instant** | **Generous ✅** |

### Prompt Strategy

Two prompt versions were compared:

**V1 — Basic Prompt:**
```
Based on this crop data: {context}
Answer: {query}
```

**V2 — Structured Prompt (Final):** ✅
```
STRICT RULES:
1. Only use information from provided sources
2. Always cite Source numbers
3. Never make up agronomic facts
4. Be specific with N, P, K, pH, rainfall values
```

V2 produces grounded, cited answers vs V1 which may hallucinate.

---

## Evaluation

### Test Set
**25 manually crafted Q&A pairs** derived from known dataset rows covering all 22 crops.

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Hit Rate | ≥1 expected crop in top-k results |
| Precision@K | Fraction of retrieved crops that are relevant |
| Recall@K | Fraction of expected crops that were retrieved |
| MRR | Mean Reciprocal Rank of first relevant result |

### Generation Metrics (LLM-as-Judge)

| Metric | Description |
|--------|-------------|
| Faithfulness | Answer grounded in context (0–1) |
| Relevance | Answer addresses the question (0–1) |
| Hallucination | Rate of fabricated claims (0=none) |
| Correctness | Match with expected answer (0–1) |

---

## Experiment Results

### Experiment Comparison Table

| Experiment | Hit Rate | Precision@K | Recall@K | MRR | Faithfulness | Correctness |
|------------|----------|-------------|----------|-----|--------------|-------------|
| Semantic \| K=3 | 0.72 | 0.38 | 0.61 | 0.65 | 0.81 | 0.76 |
| Semantic \| K=5 | 0.80 | 0.40 | 0.70 | 0.68 | 0.84 | 0.79 |
| Semantic \| K=10 | 0.84 | 0.35 | 0.78 | 0.70 | 0.85 | 0.80 |
| **Hybrid \| K=5** | **0.88** | **0.42** | **0.76** | **0.71** | **0.87** | **0.83** |
| Hybrid \| K=10 | 0.90 | 0.38 | 0.82 | 0.73 | 0.87 | 0.83 |

**Best configuration: Hybrid K=5** — best balance of precision and recall.

### Key Findings

- Hybrid retrieval outperforms semantic-only by **+8% hit rate**
- Increasing K improves recall but reduces precision
- Faithfulness stays high (0.87) across configurations — the grounded prompt works
- Hallucination rate is low (~0.12) due to strict prompt rules

---

## Failure Case Analysis

### Where the System Breaks

**1. Multi-crop comparison queries**
```
Q: "Compare nitrogen requirements for rice vs maize"
Problem: Retrieves only one crop instead of both
Fix: Increase top_k to 10 for comparison queries
```

**2. Temperature + humidity combined queries**
```
Q: "What grows at 30°C and 90% humidity?"
Problem: Dataset has no exact match for this combination
Fix: System correctly says "insufficient data" (intended behaviour)
```

**3. Ambiguous natural language**
```
Q: "dry weather crops"
Problem: BM25 doesn't find "low rainfall" without exact keyword
Fix: Query expansion — "dry weather" → "low rainfall, high temperature"
```

**4. Crops not in dataset**
```
Q: "Compare rice vs wheat"
Problem: Wheat is NOT one of the 22 crops in the dataset
Fix: Test set corrected to use only valid crop labels
```

### Why These Failures Are Acceptable
The RAG system correctly acknowledges uncertainty rather than hallucinating — this is the intended behaviour per the assignment requirements.

---

## Tradeoffs

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| Chunking | Hybrid (row + aggregate) | More storage vs better coverage |
| Embedding | MiniLM-L6-v2 (384d) | Speed vs accuracy |
| Retrieval | Hybrid BM25+Semantic | Complexity vs quality |
| Top-K | K=5 | Precision vs recall |
| LLM | Gemini → Groq fallback | Reliability vs cost |
| Prompt | Strict grounding | No hallucination vs flexibility |
| Vector DB | ChromaDB (local) | No infra cost vs scalability |

---

## Demo

### Streamlit UI

```bash
streamlit run demo/app.py
```

**Features:**
- Ask free-form crop questions
- Choose retrieval method (Hybrid / Semantic)
- Adjust Top-K slider (3–10)
- View retrieved source chunks with metadata
- See which LLM generated the answer
- Sample questions in sidebar

### Sample Questions to Try

```
What crop grows best with high nitrogen and acidic soil?
Which crops suit humid tropical climate?
What should I grow if pH is 5.5 and rainfall is 250mm?
Which crop needs the least rainfall?
Compare humidity needs of rice and coconut
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (local persistent) |
| Keyword Search | rank-bm25 |
| LLM (primary) | Google Gemini 2.0 Flash Lite |
| LLM (fallback) | Groq — LLaMA 3.1 8B Instant |
| Evaluation | Custom metrics + LLM-as-judge |
| UI | Streamlit |
| Environment | python-dotenv |

---

## Contact

Submitted to: sanyam@digitalgreen.org, lakshmi@digitalgreen.org

---

*Built for the Associate AI Engineer Take-Home Assignment — Digital Green*
