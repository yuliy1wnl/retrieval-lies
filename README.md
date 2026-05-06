# Retrieval Lies: Benchmarking Endee Vector Database

> How much do your choices of embedding model, quantization precision, and HNSW
> parameters actually affect retrieval quality? This project finds out empirically.

## Motivation

When building production retrieval systems, engineers face three decisions before
writing any application logic:

1. Which embedding model to use?
2. How aggressively to quantize vectors (float32 vs int8)?
3. Which HNSW construction parameters to set?

Conventional wisdom says float32 > int8 and higher ef_con = better recall.
This benchmark tests those assumptions against ground truth on MS MARCO.

## What This Is

A systematic retrieval benchmark evaluating **18 index configurations** across
**3 embedding models** on **25,000 MS MARCO passages** with **200 queries**
that have human-labeled relevance judgments.

This is not a demo. There is no chatbot. The output is a reproducible benchmark
report with real metrics — Precision@k, Recall@k, MRR, NDCG@k — computed against
ground truth relevance judgments.

## System Design

```
MS MARCO Dataset (HuggingFace)
        │
        ▼
┌─────────────────┐
│   data/prepare  │  Downloads 25k passages + 200 queries + qrels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│indexer/build_   │  3 models × 6 configs = 18 Endee indexes
│   indexes       │  Embeds on GPU, upserts in batches of 1000
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│evaluator/       │     │analyzer/             │
│  evaluate       │     │  failure_modes       │
│                 │     │                      │
│ P@k, R@k,       │     │ Recall@10 by query   │
│ MRR, NDCG@k     │     │ type (short/long/    │
│                 │     │ negation/technical)  │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌──────────────────┐
         │reporter/generate │  Plots + markdown report
         │    _report       │
         └──────────────────┘
```

### How Endee Is Used

Endee is the vector database. For each of the 18 configurations:

- An index is created via `client.create_index()` with specific precision and HNSW params
- 25,000 passage embeddings are upserted in batches of 1,000
- Each query embedding is searched via `index.query()` with `top_k=10, ef=128`
- Retrieved passage IDs are matched against ground truth qrels to compute metrics

Endee runs locally via Docker with no authentication (benchmark environment).

## Key Findings

### 1. Model choice dominates all other decisions

| Model                    | MRR (best config) | MRR (worst config) |
| ------------------------ | ----------------- | ------------------ |
| `BAAI/bge-small-en-v1.5` | 0.5709            | 0.5692             |
| `intfloat/e5-small-v2`   | 0.5410            | 0.5363             |
| `all-MiniLM-L6-v2`       | 0.5380            | 0.5368             |

The gap between models (~0.03 MRR) is larger than the gap between any
configuration choices within a single model. Choose your embedding model carefully —
it matters more than any index tuning.

### 2. Quantization cost is negligible

BGE-small float32 MRR: **0.5709**
BGE-small int8 MRR: **0.5692–0.5696**

A drop of 0.0013–0.0017 MRR for int8 vs float32 — essentially noise.
**Production implication:** use int8. You halve memory usage with no meaningful
retrieval quality loss.

### 3. ef_con barely matters at this corpus size

ef_con=64 and ef_con=128 produce near-identical results across all models and
precision levels. For corpora under ~100k passages, the default ef_con is
sufficient. Save the construction time.

### 4. E5 underperforms despite retrieval-specific training

`intfloat/e5-small-v2` was trained specifically for retrieval tasks yet
underperforms `BAAI/bge-small-en-v1.5` by ~0.03 MRR and ties with `all-MiniLM-L6-v2`
which was trained for general semantic similarity. Model size and training data
likely explain this — BGE models are known to be well-optimized for MS MARCO-style
retrieval.

### 5. Query length does not drive failures

All three models achieve ~95% Recall@10 on both short (<5 words) and long (≥5 words)
queries. Query length is not a meaningful failure predictor for these models on
this dataset.

**Caveat:** The negation and technical query categories contained only 2 queries
each in this sample — insufficient to draw conclusions. A larger query sample
with deliberate category stratification would be needed to study these failure modes.

## Full Results

| Model             | Precision | ef_con | MRR    | NDCG@10 |
| ----------------- | --------- | ------ | ------ | ------- |
| bge-small-en-v1.5 | float32   | 128    | 0.5709 | 0.6610  |
| bge-small-en-v1.5 | float32   | 64     | 0.5709 | 0.6609  |
| bge-small-en-v1.5 | int16     | 64     | 0.5705 | 0.6607  |
| bge-small-en-v1.5 | int16     | 128    | 0.5704 | 0.6596  |
| bge-small-en-v1.5 | int8      | 64     | 0.5696 | 0.6599  |
| bge-small-en-v1.5 | int8      | 128    | 0.5692 | 0.6584  |
| e5-small-v2       | float32   | 128    | 0.5410 | 0.6362  |
| e5-small-v2       | int16     | 128    | 0.5393 | 0.6360  |
| e5-small-v2       | int8      | 128    | 0.5387 | 0.6353  |
| all-MiniLM-L6-v2  | int8      | 64     | 0.5380 | 0.6379  |
| all-MiniLM-L6-v2  | float32   | 64     | 0.5377 | 0.6375  |
| all-MiniLM-L6-v2  | int16     | 64     | 0.5377 | 0.6375  |
| e5-small-v2       | int8      | 64     | 0.5373 | 0.6341  |
| e5-small-v2       | int16     | 64     | 0.5372 | 0.6314  |
| all-MiniLM-L6-v2  | int8      | 128    | 0.5371 | 0.6372  |
| all-MiniLM-L6-v2  | float32   | 128    | 0.5368 | 0.6370  |
| all-MiniLM-L6-v2  | int16     | 128    | 0.5368 | 0.6370  |
| e5-small-v2       | float32   | 64     | 0.5363 | 0.6312  |

## Setup & Execution

### Prerequisites

- Docker Desktop running
- Python 3.12+
- NVIDIA GPU with CUDA (CPU works but is slower)

### 1. Clone and install

```bash
git clone https://github.com/yuliy1wnl/retrieval-lies
cd retrieval-lies
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas sentence-transformers==3.3.1 transformers==4.47.0 datasets==2.21.0 pyarrow==17.0.0 tqdm requests matplotlib seaborn tabulate endee
```

### 2. Start Endee

```bash
docker compose up -d
```

Verify: `curl http://localhost:8080` should return the Endee dashboard HTML.

### 3. Run the full benchmark

```bash
python run_benchmark.py
```

This runs all 4 steps sequentially:

- **Step 1:** Download and preprocess MS MARCO (~1GB, one-time)
- **Step 2:** Build 18 Endee indexes (~20-40 min depending on hardware)
- **Step 3:** Run evaluation (Precision@k, Recall@k, MRR, NDCG@k)
- **Step 4:** Analyze failure modes by query type
- **Report:** Generate plots and `reports/report.md`

### Skip steps on re-runs

```bash
python run_benchmark.py --skip-data            # skip download
python run_benchmark.py --skip-data --skip-index  # only eval + report
```

### Output

```
results/
  index_manifest.json   — all 18 index configs
  metrics.json          — full per-config metrics
  failure_modes.json    — recall@10 by query type
reports/
  report.md             — full benchmark report
  figures/              — PNG charts
```

## Hardware Used

- GPU: NVIDIA RTX 4060 (8GB VRAM)
- CPU: AMD Ryzen 5 7600X
- RAM: 16GB
- OS: Windows 11
- Vector DB: Endee (Docker, local, no auth)

## Limitations

- 25,000 passages is a small corpus. Results at 1M+ passages may differ,
  particularly for ef_con sensitivity.
- All three embedding models are 384-dim small models. Larger models
  (768-dim, 1024-dim) may show different precision sensitivity curves.
- Negation and technical query failure analysis is inconclusive at n=2.
  Treat those numbers as directional only.

## License

MIT
