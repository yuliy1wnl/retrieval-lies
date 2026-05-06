"""
For each index in the manifest:
  - Embed each query
  - Search Endee (top-k)
  - Compute Precision@k, Recall@k, MRR, NDCG@k
Saves results/metrics.json
"""

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from endee import Endee
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    BATCH_SIZE, DATA_DIR, ENDEE_API_TOKEN, ENDEE_BASE_URL,
    EF_SEARCH, TOP_K_VALUES,
)


def load_queries(data_dir: str) -> list[dict]:
    with open(Path(data_dir) / "queries.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_qrels(data_dir: str) -> dict[str, set[str]]:
    """Returns {query_id: set of relevant passage_ids}"""
    qrels = defaultdict(set)
    with open(Path(data_dir) / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] > 0:
                qrels[r["query_id"]].add(r["passage_id"])
    return dict(qrels)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for pid in retrieved_k if pid in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for pid in retrieved_k if pid in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_index(
    index,
    queries: list[dict],
    qrels: dict,
    model: SentenceTransformer,
    max_k: int,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    query_texts = [q["text"] for q in queries]
    query_ids   = [q["id"]   for q in queries]

    # Embed all queries
    q_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    per_query_metrics = []

    for qid, qvec in tqdm(zip(query_ids, q_embeddings), total=len(query_ids), leave=False):
        relevant = qrels.get(qid, set())
        if not relevant:
            continue

        results = index.query(
            vector=qvec,
            top_k=max_k,
            ef=EF_SEARCH,
            include_vectors=False,
        )
        retrieved_ids = [r["id"] for r in results]

        row = {"query_id": qid}
        for k in TOP_K_VALUES:
            row[f"P@{k}"]    = precision_at_k(retrieved_ids, relevant, k)
            row[f"R@{k}"]    = recall_at_k(retrieved_ids, relevant, k)
            row[f"NDCG@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
        row["MRR"] = mrr(retrieved_ids, relevant)
        per_query_metrics.append(row)

    # Aggregate
    agg = {}
    for k in TOP_K_VALUES:
        agg[f"P@{k}"]    = sum(r[f"P@{k}"]    for r in per_query_metrics) / len(per_query_metrics)
        agg[f"R@{k}"]    = sum(r[f"R@{k}"]    for r in per_query_metrics) / len(per_query_metrics)
        agg[f"NDCG@{k}"] = sum(r[f"NDCG@{k}"] for r in per_query_metrics) / len(per_query_metrics)
    agg["MRR"] = sum(r["MRR"] for r in per_query_metrics) / len(per_query_metrics)
    agg["num_queries"] = len(per_query_metrics)

    return agg


def run_evaluation():
    queries = load_queries(DATA_DIR)
    qrels   = load_qrels(DATA_DIR)
    max_k   = max(TOP_K_VALUES)

    with open("results/index_manifest.json") as f:
        manifest = json.load(f)

    client = Endee(ENDEE_API_TOKEN)
    client.set_base_url(ENDEE_BASE_URL)

    all_results = []
    current_model_name = None
    model = None

    for entry in manifest:
        model_name = entry["model"]
        index_name = entry["index_name"]
        cfg        = entry["config"]

        # Only reload model when it changes
        if model_name != current_model_name:
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"\nLoading model: {model_name}")
            model = SentenceTransformer(model_name)
            current_model_name = model_name

        print(f"  Evaluating: {index_name}")
        index = client.get_index(name=index_name)

        t0 = time.time()
        metrics = evaluate_index(index, queries, qrels, model, max_k)
        elapsed = time.time() - t0

        result = {
            "model":      model_name,
            "index_name": index_name,
            "config":     cfg,
            "metrics":    metrics,
            "eval_time_s": round(elapsed, 2),
        }
        all_results.append(result)
        print(f"    MRR={metrics['MRR']:.4f}  "
              f"P@10={metrics['P@10']:.4f}  "
              f"NDCG@10={metrics['NDCG@10']:.4f}  "
              f"({elapsed:.1f}s)")

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Evaluation complete. Results saved to results/metrics.json")


if __name__ == "__main__":
    run_evaluation()