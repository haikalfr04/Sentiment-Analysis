"""Run a fine-tuned model on new reviews.

    python -m src.predict --model_dir models/distilbert-base-uncased-256-head \
        "An absolute masterpiece, I cried twice." "Two hours of my life I won't get back."
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import INFERENCE_CONFIG, clean_text, encode

DEFAULT_INFERENCE_CONFIG = {"max_length": 512, "truncation": "head"}


def load_inference_config(model_dir: str) -> dict:
    """Read the tokenization settings saved at training time (local dir or Hub repo)."""
    try:
        if os.path.isdir(model_dir):
            path = os.path.join(model_dir, INFERENCE_CONFIG)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_dir, INFERENCE_CONFIG)
        with open(path) as f:
            return {**DEFAULT_INFERENCE_CONFIG, **json.load(f)}
    except Exception:
        return dict(DEFAULT_INFERENCE_CONFIG)


class SentimentPredictor:
    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.config = load_inference_config(model_dir)

    @torch.inference_mode()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        id2label = self.model.config.id2label
        for i in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[i : i + batch_size]]
            inputs = encode(self.tokenizer, batch, self.config["max_length"], self.config["truncation"])
            inputs = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**inputs).logits, dim=-1).cpu()
            for row in probs:
                scores = {id2label[j]: float(p) for j, p in enumerate(row)}
                label = max(scores, key=scores.get)
                results.append({"label": label, "score": scores[label], "scores": scores})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model_dir", required=True, help="local directory or Hugging Face Hub repo id")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir)
    for text, result in zip(args.texts, predictor.predict(args.texts)):
        print(f"{result['label']:>8} ({result['score']:.3f})  {text}")


if __name__ == "__main__":
    main()
