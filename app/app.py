"""Gradio demo for the fine-tuned movie review sentiment classifier.

    MODEL_DIR=models/distilbert-base-uncased-256-head python app/app.py

MODEL_DIR can also be a Hugging Face Hub repo id, e.g. "<username>/imdb-distilbert".
"""

import os
import sys

import gradio as gr

# Find src/ both when run from this repo (app/app.py) and on a Hugging Face Space (app.py at the root).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]
from src.predict import SentimentPredictor  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "models/distilbert-base-uncased-256-head")

EXAMPLES = [
    "One of the best films I have seen in years. The performances are stunning and the score gave me chills.",
    "I wanted to love this, but the plot made no sense and the ending was a complete letdown.",
    "Oh great, another two hours of explosions and zero character development. Just what cinema needed.",
    "The first half drags badly, but the last forty minutes are some of the most gripping I've ever watched.",
]

predictor = SentimentPredictor(MODEL_DIR)


def classify(review: str) -> dict:
    if not review.strip():
        return {}
    return predictor.predict([review])[0]["scores"]


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=6, label="Movie review", placeholder="Paste a movie review..."),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    examples=EXAMPLES,
    title="Movie Review Sentiment Analysis",
    description=(
        "A transformer fine-tuned on 25,000 IMDB reviews classifies a review as positive or negative. "
        "Try sarcasm or mixed reviews to find where it breaks."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
