"""
Classical baseline: TF-IDF + logistic regression on the same PubMed 20k RCT
train/test split the transformer uses. Gives the DistilBERT result a floor to
be measured against.

    python -m scripts.baseline

Fits the pipeline, logs a run to the `clinical-text-classifier` MLflow
experiment, saves model_output/baseline.joblib, and writes docs/baseline.md.
"""

from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.data.loader import load_dataset

TRACKING_URI = f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}"
MODEL_OUT = Path("model_output/baseline.joblib")

# DistilBERT test numbers, from `python -m scripts.evaluate`.
DISTILBERT_ACC = 0.866
DISTILBERT_F1 = 0.806


def main():
    bundle = load_dataset("pubmed_20k_rct")
    labels = bundle.label_names
    tc, lc = bundle.text_column, bundle.label_column

    x_train = list(bundle.dataset["train"][tc])
    y_train = np.array(bundle.dataset["train"][lc])
    x_test = list(bundle.dataset["test"][tc])
    y_test = np.array(bundle.dataset["test"][lc])

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    print("fitting TF-IDF + logistic regression ...")
    pipe.fit(x_train, y_train)

    pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro")
    rep = classification_report(y_test, pred, target_names=labels, output_dict=True)
    n_feat = len(pipe.named_steps["tfidf"].vocabulary_)

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("clinical-text-classifier")
    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.log_params(
            {
                "model_type": "tfidf+logreg",
                "ngram_range": "(1, 2)",
                "min_df": 2,
                "max_features": 100_000,
                "n_features_used": n_feat,
                "C": 1.0,
            }
        )
        mlflow.log_metrics({"test_accuracy": acc, "test_f1_macro": f1_macro})

    lines = [
        "# Baseline: TF-IDF + logistic regression\n",
        "A deliberately non-neural model on the same train/test split, so the "
        "DistilBERT numbers have a floor to be compared against. Bag of "
        "unigrams + bigrams, TF-IDF weighted, into a linear classifier.\n",
        "This file is generated. Re-run with `python -m scripts.baseline`.\n",
        "## Test set\n",
        "| model | accuracy | macro F1 |",
        "| --- | --- | --- |",
        f"| TF-IDF + logistic regression | {acc:.3f} | {f1_macro:.3f} |",
        f"| DistilBERT (single sentence) | {DISTILBERT_ACC:.3f} | {DISTILBERT_F1:.3f} |",
        f"| gain from the transformer | +{DISTILBERT_ACC - acc:.3f} | "
        f"+{DISTILBERT_F1 - f1_macro:.3f} |\n",
        "## Per-class (baseline)\n",
        "| class | precision | recall | f1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in labels:
        r = rep[name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1-score']:.3f} | {int(r['support']):,} |"
        )
    lines += [
        "",
        "## What this tells me\n",
        f"- A bag of words already gets ~{acc:.0%}. Most of this task is just "
        'which words show up — numbers and "p <" point at `results`, '
        '"we conclude" / "these findings" point at `conclusions`.',
        f"- DistilBERT adds ~{DISTILBERT_ACC - acc:.0%} accuracy on top. That's "
        "the part that needs word order and phrasing rather than word counts.",
        "- Both models are weakest on the same classes (`objective`, "
        "`background`), so that gap is about the task being genuinely "
        "ambiguous, not about the model.",
        "",
    ]
    Path("docs/baseline.md").write_text("\n".join(lines))

    print(f"\nbaseline   acc={acc:.3f}  macro_f1={f1_macro:.3f}  ({n_feat:,} features)")
    print(f"distilbert acc={DISTILBERT_ACC:.3f}  macro_f1={DISTILBERT_F1:.3f}")
    print(f"wrote docs/baseline.md and {MODEL_OUT}")


if __name__ == "__main__":
    main()
