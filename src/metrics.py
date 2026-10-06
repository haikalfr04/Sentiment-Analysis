"""Metrics and result files shared by every model, so runs are directly comparable."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_run(
    run_name: str,
    metrics: dict,
    texts: list[str],
    labels,
    preds,
    prob_positive,
    results_dir: str = "results",
) -> None:
    """Write `<run_name>_metrics.json` and `<run_name>_predictions.csv` to `results_dir`."""
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{run_name}_metrics.json"), "w") as f:
        json.dump({"run_name": run_name, **metrics}, f, indent=2)

    pd.DataFrame(
        {
            "text": texts,
            "label": np.asarray(labels),
            "pred": np.asarray(preds),
            "prob_positive": np.asarray(prob_positive),
        }
    ).to_csv(os.path.join(results_dir, f"{run_name}_predictions.csv"), index=False)
    print(f"[{run_name}] test metrics: {json.dumps(metrics, indent=2)}")
