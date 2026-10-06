"""Classical baseline: TF-IDF (word 1-2 grams) + Logistic Regression.

The regularization strength C is picked on the validation split, then the model
is evaluated once on the test split.

    python -m src.baseline
"""

import argparse
import os
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data import DATASET_NAME, load_imdb
from src.metrics import classification_metrics, save_run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--c_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--max_features", type=int, default=200_000)
    parser.add_argument("--output_dir", default="models/tfidf-logreg")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--run_name", default="tfidf-logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def build_pipeline(c: float, max_features: int):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True),
        LogisticRegression(C=c, max_iter=2000),
    )


def main():
    args = parse_args()
    ds = load_imdb(
        args.dataset,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    train, val, test = ds["train"], ds["validation"], ds["test"]

    best_c, best_acc, best_model, best_train_time = None, -1.0, None, None
    for c in args.c_grid:
        start = time.perf_counter()
        model = build_pipeline(c, args.max_features).fit(train["text"], train["label"])
        train_time = time.perf_counter() - start
        acc = classification_metrics(val["label"], model.predict(val["text"]))["accuracy"]
        print(f"C={c:<5} val_accuracy={acc:.4f}")
        if acc > best_acc:
            best_c, best_acc, best_model, best_train_time = c, acc, model, train_time

    start = time.perf_counter()
    prob_positive = best_model.predict_proba(test["text"])[:, 1]
    inference_time = time.perf_counter() - start
    preds = (prob_positive >= 0.5).astype(int)

    metrics = classification_metrics(test["label"], preds)
    metrics.update(
        model="TF-IDF + LogisticRegression",
        best_C=best_c,
        val_accuracy=best_acc,
        num_parameters=int(best_model[-1].coef_.size + best_model[-1].intercept_.size),
        train_time_sec=round(best_train_time, 1),
        inference_samples_per_sec=round(len(test) / inference_time, 1),
    )
    save_run(args.run_name, metrics, test["text"], test["label"], preds, prob_positive, args.results_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(args.output_dir, "model.joblib"))


if __name__ == "__main__":
    main()
