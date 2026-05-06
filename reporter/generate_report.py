"""
Reads results/metrics.json and results/failure_modes.json
Produces:
  reports/figures/  — PNG charts
  reports/report.md — Full benchmark report
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
FIGS_DIR    = REPORTS_DIR / "figures"


def load_metrics() -> pd.DataFrame:
    with open(RESULTS_DIR / "metrics.json") as f:
        data = json.load(f)
    rows = []
    for entry in data:
        row = {
            "model":      entry["model"].split("/")[-1],
            "precision":  entry["config"]["precision"],
            "ef_con": entry["config"]["ef_con"],
            "index_name": entry["index_name"],
        }
        row.update(entry["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_failure_modes() -> dict:
    with open(RESULTS_DIR / "failure_modes.json") as f:
        return json.load(f)


def setup():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def plot_mrr_by_model_precision(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(index="model", columns="precision", values="MRR", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Mean Reciprocal Rank by Model × Precision", fontweight="bold")
    ax.set_ylabel("MRR")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="Precision")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "mrr_model_precision.png")
    plt.close()


def plot_ndcg_at_k(df: pd.DataFrame):
    k_cols = ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].groupby("precision")[k_cols].mean()
        sub.T.plot(ax=ax, marker="o")
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel("NDCG@k")
        ax.set_ylim(0, 1)
        ax.legend(title="Precision", fontsize=8)
    fig.suptitle("NDCG@k across Precision Levels", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "ndcg_at_k.png", bbox_inches="tight")
    plt.close()


def plot_precision_recall(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric_prefix, label in zip(
        axes, ["P@", "R@"], ["Precision@k", "Recall@k"]
    ):
        k_cols = [f"{metric_prefix}{k}" for k in [1, 3, 5, 10]]
        agg = df.groupby("model")[k_cols].mean().T
        agg.index = [1, 3, 5, 10]
        agg.plot(ax=ax, marker="o")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.legend(title="Model", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "precision_recall_at_k.png")
    plt.close()


def plot_failure_modes(failure_data: dict):
    categories = ["short", "long", "negation", "technical", "general"]
    models = list(failure_data.keys())
    model_labels = [m.split("/")[-1] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, (model, label) in enumerate(zip(models, model_labels)):
        vals = [
            failure_data[model].get(cat, {}).get("recall@10", 0)
            for cat in categories
        ]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_title("Recall@10 by Query Type (Failure Mode Analysis)", fontweight="bold")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend(title="Model")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "failure_modes.png")
    plt.close()


def build_summary_table(df: pd.DataFrame) -> str:
    cols = ["model", "precision", "ef_con", "MRR", "P@10", "R@10", "NDCG@10"]
    summary = df[cols].sort_values("MRR", ascending=False).reset_index(drop=True)
    summary["MRR"]     = summary["MRR"].map("{:.4f}".format)
    summary["P@10"]    = summary["P@10"].map("{:.4f}".format)
    summary["R@10"]    = summary["R@10"].map("{:.4f}".format)
    summary["NDCG@10"] = summary["NDCG@10"].map("{:.4f}".format)
    return summary.to_markdown(index=False)


def generate_report():
    setup()
    df = load_metrics()
    failure_data = load_failure_modes()

    plot_mrr_by_model_precision(df)
    plot_ndcg_at_k(df)
    plot_precision_recall(df)
    plot_failure_modes(failure_data)

    best = df.loc[df["MRR"].idxmax()]
    worst = df.loc[df["MRR"].idxmin()]

    report = f"""# Retrieval Lies: Benchmarking Endee Vector Database

## Overview
This benchmark evaluates [Endee](https://endee.io) across **{len(df)}** index configurations,
combining **{df['model'].nunique()} embedding models** × **{len(df) // df['model'].nunique()} index configs**
on **{int(df['num_queries'].max())} MS MARCO queries** against a corpus of 25,000 passages.

**Key question:** How much does your choice of embedding model, quantization precision,
and HNSW construction parameters actually affect retrieval quality?

---

## Results Summary

{build_summary_table(df)}

---

## Key Findings

### Best Configuration
- **Model:** `{best['model']}`
- **Precision:** `{best['precision']}`
- **EF Construction:** `{best['ef_con']}`
- **MRR:** `{float(best['MRR']):.4f}` | **NDCG@10:** `{float(best['NDCG@10']):.4f}`

### Worst Configuration
- **Model:** `{worst['model']}`
- **Precision:** `{worst['precision']}`
- **EF Construction:** `{worst['ef_con']}`
- **MRR:** `{float(worst['MRR']):.4f}` | **NDCG@10:** `{float(worst['NDCG@10']):.4f}`

### Precision Impact
The gap between `float32` and `int8` quantization across all models
shows the real cost of aggressive compression on retrieval quality.

---

## Charts

### MRR by Model × Precision
![MRR](figures/mrr_model_precision.png)

### NDCG@k Curves
![NDCG](figures/ndcg_at_k.png)

### Precision & Recall @k
![PR](figures/precision_recall_at_k.png)

### Failure Mode Analysis
![Failures](figures/failure_modes.png)

---

## Methodology

- **Dataset:** MS MARCO Passage Ranking (25,000 passages, 200 queries with ground truth)
- **Embedding models:** `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`
- **Index configs:** precision ∈ {{float32, int16, int8}} × ef_con ∈ {{64, 128}}
- **Metrics:** Precision@k, Recall@k, MRR, NDCG@k for k ∈ {{1, 3, 5, 10}}
- **Hardware:** NVIDIA RTX 4060, Ryzen 5 7600X
- **Vector DB:** Endee (Docker, local)

---

*Generated by [retrieval-lies](https://github.com/YOUR_USERNAME/retrieval-lies)*
"""

    with open(REPORTS_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ Report saved to reports/report.md")
    print("✅ Figures saved to reports/figures/")


if __name__ == "__main__":
    generate_report()