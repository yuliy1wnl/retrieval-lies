"""
Breaks down retrieval failures by query type:
  - Short queries (< 5 words)
  - Long queries (>= 5 words)
  - Queries containing negation words
  - Queries with rare/technical terms
Saves results/failure_modes.json
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from endee import Endee
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    BATCH_SIZE, DATA_DIR, EMBEDDING_MODELS,
    ENDEE_API_TOKEN, ENDEE_BASE_URL, EF_SEARCH,
)

NEGATION_WORDS = {"not", "no", "never", "without", "except", "neither", "nor"}

BEST_CONFIG_SUFFIX = "float32_ef128"  # evaluate failure modes on best config only


def classify_query(text: str) -> list[str]:
    words = text.lower().split()
    tags = []
    tags.append("short" if len(words) < 5 else "long")
    if any(w in NEGATION_WORDS for w in words):
        tags.append("negation")
    # Rough heuristic for technical: contains numbers or camelCase or acronyms
    if re.search(r'\d|[A-Z]{2,}|[a-z][A-Z]', text):
        tags.append("technical")
    else:
        tags.append("general")
    return tags


def run_failure_analysis():
    with open(Path(DATA_DIR) / "queries.jsonl", encoding="utf-8") as f:
        queries = [json.loads(l) for l in f]

    qrels = defaultdict(set)
    with open(Path(DATA_DIR) / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] > 0:
                qrels[r["query_id"]].add(r["passage_id"])

    client = Endee(ENDEE_API_TOKEN)
    client.set_base_url(ENDEE_BASE_URL)

    analysis = {}

    for model_name in EMBEDDING_MODELS:
        model_slug = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
        index_name = f"{model_slug}_{BEST_CONFIG_SUFFIX}"

        print(f"\nAnalyzing failure modes: {index_name}")
        model = SentenceTransformer(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        index = client.get_index(name=index_name)

        # Group metrics by query category
        category_results = defaultdict(lambda: {"hits": 0, "total": 0})

        query_texts = [q["text"] for q in queries]
        query_ids   = [q["id"]   for q in queries]

        embeddings = model.encode(
            query_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        for qid, qvec, qtext in tqdm(
            zip(query_ids, embeddings, query_texts), total=len(queries)
        ):
            relevant = qrels.get(qid, set())
            if not relevant:
                continue

            results = index.query(vector=qvec, top_k=10, ef=EF_SEARCH, include_vectors=False)
            retrieved_ids = {r["id"] for r in results}
            hit = bool(retrieved_ids & relevant)

            tags = classify_query(qtext)
            for tag in tags:
                category_results[tag]["total"] += 1
                category_results[tag]["hits"]  += int(hit)

        # Compute recall@10 per category
        model_analysis = {}
        for cat, counts in category_results.items():
            model_analysis[cat] = {
                "recall@10":   counts["hits"] / counts["total"] if counts["total"] else 0,
                "total_queries": counts["total"],
                "hits":          counts["hits"],
            }

        analysis[model_name] = model_analysis
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open("results/failure_modes.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\n✅ Failure mode analysis saved to results/failure_modes.json")


if __name__ == "__main__":
    run_failure_analysis()