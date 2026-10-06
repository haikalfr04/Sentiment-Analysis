"""Exploratory data analysis: class balance and review length.

The length statistics motivate the truncation experiments: every review longer
than `max_length` tokens loses information, so it matters which part is kept.

    python -m src.eda --tokenizer distilbert-base-uncased
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.data import DATASET_NAME, LABELS, load_imdb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--tokenizer", default="distilbert-base-uncased")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    ds = load_imdb(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    df = ds["train"].to_pandas()
    df["sentiment"] = df["label"].map(dict(enumerate(LABELS)))
    df["n_words"] = df["text"].str.split().str.len()
    df["n_tokens"] = [len(ids) for ids in tokenizer(df["text"].tolist(), verbose=False)["input_ids"]]

    stats = {
        "split_sizes": {name: len(split) for name, split in ds.items()},
        "train_label_counts": df["sentiment"].value_counts().to_dict(),
        "tokens": {k: float(v) for k, v in df["n_tokens"].describe(percentiles=[0.5, 0.9, 0.95]).items()},
        "pct_longer_than": {n: round(float((df["n_tokens"] > n).mean() * 100), 1) for n in (128, 256, 512)},
        "mean_tokens_by_sentiment": df.groupby("sentiment")["n_tokens"].mean().round(1).to_dict(),
    }
    print(json.dumps(stats, indent=2))

    figures_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "eda_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, np.percentile(df["n_tokens"], 99), 60)
    for sentiment, group in df.groupby("sentiment"):
        ax.hist(group["n_tokens"], bins=bins, alpha=0.6, label=sentiment)
    for n in (256, 512):
        ax.axvline(n, color="black", linestyle="--", linewidth=1)
        ax.text(n, ax.get_ylim()[1] * 0.95, f" {n}", va="top")
    ax.set(
        title=f"IMDB review length ({args.tokenizer} tokens)",
        xlabel="tokens per review",
        ylabel="reviews",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "length_distribution.png"), dpi=150)
    print(f"Saved {figures_dir}/length_distribution.png")

    pd.set_option("display.max_colwidth", 200)
    print("\nSample reviews:\n", df.sample(4, random_state=0)[["sentiment", "text"]])


if __name__ == "__main__":
    main()
