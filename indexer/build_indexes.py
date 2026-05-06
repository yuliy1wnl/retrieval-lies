"""
For each (embedding_model x index_config) combination:
  1. Create an Endee index
  2. Embed all passages in batches
  3. Upsert into Endee
"""

import json
import os
import time
from pathlib import Path

import torch
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    BATCH_SIZE, DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODELS,
    ENDEE_API_TOKEN, ENDEE_BASE_URL, INDEX_CONFIGS,
)

PRECISION_MAP = {
    "float32": Precision.FLOAT32,
    "float16": Precision.FLOAT16,
    "int16":   Precision.INT16,
    "int8":    Precision.INT8,
}

def index_name(model_name: str, cfg: dict) -> str:
    model_slug = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    return f"{model_slug}_{cfg['precision']}_ef{cfg['ef_con']}"

def load_passages(data_dir: str) -> list[dict]:
    path = Path(data_dir) / "passages.jsonl"
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def embed_passages(model: SentenceTransformer, texts: list[str]) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Embedding {len(texts)} passages on {device}…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def build_all_indexes():
    passages = load_passages(DATA_DIR)
    print(f"Loaded {len(passages)} passages.\n")

    client = Endee(ENDEE_API_TOKEN)
    client.set_base_url(ENDEE_BASE_URL)

    # Track what we build for the evaluator
    built = []

    for model_name in EMBEDDING_MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name)
        texts = [p["text"] for p in passages]
        embeddings = embed_passages(model, texts)

        for cfg in INDEX_CONFIGS:
            name = index_name(model_name, cfg)
            print(f"\n  Index: {name}")

            # Delete if exists (clean re-run)
            try:
                client.delete_index(name)
                print(f"  Deleted existing index.")
            except Exception:
                pass

            # Create index
            client.create_index(
                name=name,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision=PRECISION_MAP[cfg["precision"]],
                ef_con=cfg["ef_con"],
                M=cfg["M"],
            )
            print(f"  Created index (precision={cfg['precision']}, "
                  f"ef_con={cfg['ef_con']}, M={cfg['M']})")

            # Upsert in batches of 1000 (Endee limit)
            index = client.get_index(name=name)
            batch_size = 1000
            total = len(passages)
            for start in tqdm(range(0, total, batch_size), desc="  Upserting"):
                batch_passages = passages[start:start + batch_size]
                batch_embeddings = embeddings[start:start + batch_size]
                items = [
                    {
                        "id": p["id"],
                        "vector": emb,
                        "meta": {"text": p["text"]},
                        "filter": {},
                    }
                    for p, emb in zip(batch_passages, batch_embeddings)
                ]
                index.upsert(items)

            built.append({
                "model": model_name,
                "index_name": name,
                "config": cfg,
            })
            print(f"  ✅ {name} ready.")

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save manifest for evaluator
    os.makedirs("results", exist_ok=True)
    with open("results/index_manifest.json", "w") as f:
        json.dump(built, f, indent=2)
    print(f"\n✅ All indexes built. Manifest saved to results/index_manifest.json")


if __name__ == "__main__":
    build_all_indexes()