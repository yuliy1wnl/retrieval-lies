"""
Download MS MARCO passage ranking subset and prepare ground truth.
Saves:
  data/msmarco/passages.jsonl   — {id, text}
  data/msmarco/queries.jsonl    — {id, text}
  data/msmarco/qrels.jsonl      — {query_id, passage_id, relevance}
"""

import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, NUM_PASSAGES, NUM_QUERIES


def prepare():
    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading MS MARCO from HuggingFace (this downloads ~1GB first time)…")
    ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)

    passages, queries, qrels = {}, {}, []
    passage_counter = 0

    print("Extracting passages, queries, and relevance judgments…")
    for row in tqdm(ds):
        qid = str(row["query_id"])
        q_text = row["query"].strip()

        selected_passages = []
        for i, (text, is_selected) in enumerate(
            zip(row["passages"]["passage_text"], row["passages"]["is_selected"])
        ):
            text = text.strip()
            if not text:
                continue
            pid = f"{qid}_{i}"
            if pid not in passages:
                passages[pid] = text
                passage_counter += 1
            selected_passages.append((pid, int(is_selected)))

        # Only keep queries that have at least one relevant passage
        relevant = [(pid, rel) for pid, rel in selected_passages if rel == 1]
        if not relevant:
            continue

        queries[qid] = q_text
        for pid, rel in selected_passages:
            qrels.append({"query_id": qid, "passage_id": pid, "relevance": rel})

        if len(queries) >= NUM_QUERIES and passage_counter >= NUM_PASSAGES:
            break

    # Trim to limits
    selected_qids = list(queries.keys())[:NUM_QUERIES]
    relevant_pids = {
        r["passage_id"]
        for r in qrels
        if r["query_id"] in selected_qids
    }

    # Fill up to NUM_PASSAGES with random passages (negatives)
    all_pids = list(passages.keys())
    random.seed(42)
    random.shuffle(all_pids)
    final_pids = set(relevant_pids)
    for pid in all_pids:
        if len(final_pids) >= NUM_PASSAGES:
            break
        final_pids.add(pid)

    # Write passages
    print(f"Writing {len(final_pids)} passages…")
    with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
        for pid in final_pids:
            f.write(json.dumps({"id": pid, "text": passages[pid]}) + "\n")

    # Write queries
    print(f"Writing {len(selected_qids)} queries…")
    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid in selected_qids:
            f.write(json.dumps({"id": qid, "text": queries[qid]}) + "\n")

    # Write qrels (only for selected queries and final passages)
    filtered_qrels = [
        r for r in qrels
        if r["query_id"] in selected_qids and r["passage_id"] in final_pids
    ]
    print(f"Writing {len(filtered_qrels)} relevance judgments…")
    with open(out / "qrels.jsonl", "w", encoding="utf-8") as f:
        for r in filtered_qrels:
            f.write(json.dumps(r) + "\n")

    print(f"\n✅ Done. {len(final_pids)} passages | {len(selected_qids)} queries | "
          f"{len(filtered_qrels)} qrels saved to {out}/")


if __name__ == "__main__":
    prepare()