"""
Fine-tunes DistilBERT on a clinical text classification dataset, with every
run tracked in MLflow (params, metrics, and the resulting model artifact).

Usage:
    python -m src.train --dataset pubmed_20k_rct --epochs 3 --lr 2e-5
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset

BASE_MODEL = "distilbert-base-uncased"

# Pin the MLflow tracking store to a single sqlite file at the repo root.
# Without this, MLflow's default resolves relative to the current working
# directory and gets percent-encoded when the path contains spaces, so the
# training client and `mlflow ui` can end up reading different db files.
# Anchoring to __file__ keeps every run -- and the UI -- on one store.
TRACKING_URI = f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}"


def build_tokenize_fn(tokenizer, text_column: str):
    def tokenize(batch):
        # No padding here -- we tokenize once up front, but leave every
        # sequence at its natural length. Padding is applied per-batch at
        # training time by DataCollatorWithPadding, so each batch is only
        # as wide as its longest sentence instead of a fixed 128. The
        # max_length cap still guards against a pathologically long outlier.
        return tokenizer(batch[text_column], truncation=True, max_length=128)

    return tokenize


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubmed_20k_rct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="./model_output")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, truncate every split to this many rows (fast smoke runs).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, stop after this many optimizer steps (overrides --epochs).",
    )
    args = parser.parse_args()

    bundle = load_dataset(args.dataset)
    if args.limit > 0:
        bundle.dataset = DatasetDict(
            {split: ds.select(range(min(args.limit, len(ds))))
             for split, ds in bundle.dataset.items()}
        )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized = bundle.dataset.map(
        build_tokenize_fn(tokenizer, bundle.text_column), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(bundle.label_names),
        id2label=dict(enumerate(bundle.label_names)),
        label2id={name: i for i, name in enumerate(bundle.label_names)},
    )

    val_split = "validation" if "validation" in tokenized else "val"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[val_split],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("clinical-text-classifier")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "dataset": bundle.source_name,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_labels": len(bundle.label_names),
            }
        )

        trainer.train()

        test_metrics = trainer.evaluate(tokenized["test"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        # The API reads this to map class indices -> human-readable names.
        with open(f"{args.output_dir}/final/label_names.json", "w") as f:
            json.dump(bundle.label_names, f)
        mlflow.log_artifacts(f"{args.output_dir}/final", artifact_path="model")

        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
