"""Compare all runs in `results/` and analyse their errors.

Produces:
  results/comparison.md                 - metrics table across every run
  results/figures/<run>_confusion.png   - confusion matrix per run
  results/errors/<run>_errors.md        - the most confident mistakes per run

    python -m src.evaluate
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data import LABELS

TABLE_COLUMNS = [
    ("run_name", "Run"),
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("num_parameters", "Params"),
    ("train_time_sec", "Train time"),
    ("inference_samples_per_sec", "Inference (samples/s)"),
]


def _format(key, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if key in ("accuracy", "f1"):
        return f"{value * 100:.2f}%"
    if key == "num_parameters":
        return f"{value / 1e6:.1f}M"
    if key == "train_time_sec":
        return f"{value / 60:.1f} min"
    return str(value)


def comparison_table(results_dir: str) -> str:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    runs.sort(key=lambda r: r["accuracy"])

    header = "| " + " | ".join(title for _, title in TABLE_COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    rows = ["| " + " | ".join(_format(key, run.get(key)) for key, _ in TABLE_COLUMNS) + " |" for run in runs]
    return "\n".join([header, divider, *rows])


def plot_confusion(df: pd.DataFrame, run_name: str, path: str) -> None:
    cm = confusion_matrix(df["label"], df["pred"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(title=run_name, xlabel="predicted", ylabel="actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_report(df: pd.DataFrame, run_name: str, top_k: int) -> str:
    """Most confident mistakes: where the model was sure and still wrong."""
    errors = df[df["label"] != df["pred"]].copy()
    errors["confidence"] = errors["prob_positive"].where(errors["pred"] == 1, 1 - errors["prob_positive"])

    lines = [
        f"# Error analysis: {run_name}",
        "",
        f"{len(errors):,} of {len(df):,} test reviews misclassified ({len(errors) / len(df):.2%}).",
        "",
        f"- False positives (negative review predicted positive): {int((errors['pred'] == 1).sum()):,}",
        f"- False negatives (positive review predicted negative): {int((errors['pred'] == 0).sum()):,}",
        "",
        "Typical causes to look for: sarcasm, mixed reviews (\"great acting, terrible plot\"),"
        " negation, and verdicts that only appear at the end of a long review.",
    ]
    for pred, title in ((1, "False positives"), (0, "False negatives")):
        lines += ["", f"## Most confident {title.lower()}", ""]
        for _, row in errors[errors["pred"] == pred].nlargest(top_k, "confidence").iterrows():
            text = row["text"] if len(row["text"]) <= 600 else row["text"][:600] + " ..."
            lines.append(f"- **{row['confidence']:.3f}** {text}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    table = comparison_table(args.results_dir)
    with open(os.path.join(args.results_dir, "comparison.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    figures_dir = os.path.join(args.results_dir, "figures")
    errors_dir = os.path.join(args.results_dir, "errors")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_predictions.csv"))):
        run_name = os.path.basename(path).removesuffix("_predictions.csv")
        df = pd.read_csv(path)
        plot_confusion(df, run_name, os.path.join(figures_dir, f"{run_name}_confusion.png"))
        with open(os.path.join(errors_dir, f"{run_name}_errors.md"), "w") as f:
            f.write(error_report(df, run_name, args.top_k))
        print(f"[{run_name}] confusion matrix and error report written")


if __name__ == "__main__":
    main()
