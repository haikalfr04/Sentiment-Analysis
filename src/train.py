"""Fine-tune a pretrained transformer (BERT, DistilBERT, RoBERTa, ...) on IMDB.

    # quick, Colab-free-tier friendly
    python -m src.train --model_name distilbert-base-uncased

    # strongest setup: longer context with head+tail truncation
    python -m src.train --model_name roberta-base --max_length 512 --truncation head_tail

    # smoke test on a tiny subset
    python -m src.train --max_train_samples 500 --max_eval_samples 200 --epochs 1
"""

import argparse
import json
import math
import glob
import os
import shutil
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import DATASET_NAME, ID2LABEL, INFERENCE_CONFIG, LABEL2ID, load_imdb, tokenize_dataset
from src.metrics import classification_metrics, save_run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--truncation", choices=["head", "head_tail"], default="head")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--run_name", default=None, help="defaults to <model>-<max_length>-<truncation>")
    parser.add_argument("--output_dir", default=None, help="defaults to models/<run_name>")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--report_to", default="none", help='"none", "wandb", "tensorboard", ...')
    parser.add_argument("--hub_model_id", default=None, help="push the final model to this Hugging Face Hub repo")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return classification_metrics(labels, np.argmax(logits, axis=-1))


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"{args.model_name.split('/')[-1]}-{args.max_length}-{args.truncation}"
    output_dir = args.output_dir or os.path.join("models", run_name)

    ds = load_imdb(
        args.dataset,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    test_texts, test_labels = ds["test"]["text"], ds["test"]["label"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_dataset(ds, tokenizer, args.max_length, args.truncation)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )

    steps_per_epoch = math.ceil(len(tokenized["train"]) / (args.batch_size * args.grad_accum))
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_steps=int(steps_per_epoch * args.epochs * args.warmup_ratio),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start
    val_metrics = trainer.evaluate()

    start = time.perf_counter()
    logits = trainer.predict(tokenized["test"]).predictions
    inference_time = time.perf_counter() - start
    prob_positive = torch.softmax(torch.from_numpy(logits).float(), dim=-1)[:, 1].numpy()
    preds = logits.argmax(axis=-1)

    metrics = classification_metrics(test_labels, preds)
    metrics.update(
        model=args.model_name,
        max_length=args.max_length,
        truncation=args.truncation,
        epochs=args.epochs,
        lr=args.lr,
        val_accuracy=val_metrics["eval_accuracy"],
        num_parameters=sum(p.numel() for p in model.parameters()),
        train_time_sec=round(train_time, 1),
        inference_samples_per_sec=round(len(test_labels) / inference_time, 1),
        device=torch.cuda.get_device_name(0) if use_cuda else "cpu",
    )
    save_run(run_name, metrics, test_texts, test_labels, preds, prob_positive, args.results_dir)

    # Keep only the best model; the predictor and the demo app read the
    # inference config so they tokenize exactly like training did.
    trainer.save_model(output_dir)
    for checkpoint in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint)
    with open(os.path.join(output_dir, INFERENCE_CONFIG), "w") as f:
        json.dump({"max_length": args.max_length, "truncation": args.truncation}, f, indent=2)

    if args.hub_model_id:
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=os.path.join(output_dir, INFERENCE_CONFIG),
            path_in_repo=INFERENCE_CONFIG,
            repo_id=args.hub_model_id,
        )
        print(f"Pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
