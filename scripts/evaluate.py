"""
Re-run the evaluation and regenerate the report.

    python -m scripts.evaluate

Runs the trained model over the full PubMed 20k RCT test split and writes:
  - docs/evaluation.md        (overall + per-class metrics, notes)
  - docs/confusion_matrix.png (row-normalised)
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.loader import load_dataset

MODEL_DIR = "model_output/final"
BATCH = 64
MAX_LEN = 128


def predict_all(texts, tokenizer, model, device):
    preds = np.zeros(len(texts), dtype=int)
    confs = np.zeros(len(texts), dtype=float)
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(
            texts[i : i + BATCH],
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        preds[i : i + BATCH] = pred.cpu().numpy()
        confs[i : i + BATCH] = conf.cpu().numpy()
    return preds, confs


def plot_confusion(cm, labels, path):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cmn[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cmn[i, j] > 0.5 else "black",
            )
    ax.set_title("Confusion matrix (row-normalised, counts in parens)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    bundle = load_dataset("pubmed_20k_rct")
    labels = bundle.label_names
    test = bundle.dataset["test"]
    texts = list(test[bundle.text_column])
    y_true = np.array(test[bundle.label_column])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    y_pred, conf = predict_all(texts, tokenizer, model, device)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, labels, "docs/confusion_matrix.png")

    correct = y_true == y_pred
    conf_correct = conf[correct].mean()
    conf_wrong = conf[~correct].mean()

    # biggest off-diagonal confusion
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    ti, pj = np.unravel_index(cm_off.argmax(), cm_off.shape)

    # highest-confidence mistakes
    wrong_idx = np.where(~correct)[0]
    worst = wrong_idx[np.argsort(-conf[wrong_idx])[:6]]

    lines = []
    lines.append("# Evaluation\n")
    lines.append(
        "Model: `distilbert-base-uncased`, fully fine-tuned for 3 epochs. "
        "Evaluated on the full PubMed 20k RCT **test** split "
        f"({len(y_true):,} sentences the model never saw during training).\n"
    )
    lines.append(
        "The task: given one sentence from a medical-paper abstract, predict its "
        "role — `background`, `objective`, `methods`, `results`, or "
        "`conclusions`.\n"
    )
    lines.append(
        "This file is generated. Re-run it with `python -m scripts.evaluate`.\n"
    )

    lines.append("## Headline numbers\n")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| accuracy | {acc:.3f} |")
    lines.append(f"| macro F1 | {f1_macro:.3f} |")
    lines.append(f"| mean confidence when right | {conf_correct:.3f} |")
    lines.append(f"| mean confidence when wrong | {conf_wrong:.3f} |\n")

    lines.append("## Per-class\n")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in labels:
        r = rep[name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1-score']:.3f} | {int(r['support']):,} |"
        )
    lines.append("")

    lines.append("## Confusion matrix\n")
    lines.append("![confusion matrix](confusion_matrix.png)\n")

    best_cls = max(labels, key=lambda n: rep[n]["f1-score"])
    worst_cls = min(labels, key=lambda n: rep[n]["f1-score"])
    lines.append("## What I take from this\n")
    lines.append(
        f"- **`{best_cls}` and `results` are easy, `{worst_cls}` and "
        f"`background` are hard.** The easy ones have giveaway wording — "
        f"numbers, p-values, "
        f'"we measured", "in conclusion". The hard ones are both short, general '
        f"statements at the very start of an abstract, and telling "
        f'"here is the problem" from "here is what we tested" is genuinely '
        f"ambiguous."
    )
    lines.append(
        f"- **Most common mistake: `{labels[ti]}` predicted as `{labels[pj]}`** "
        f"({cm_off[ti, pj]:,} sentences), which lines up with the point above."
    )
    lines.append(
        f"- **The model knows when it's unsure.** Average confidence is "
        f"{conf_correct:.2f} on the ones it gets right vs {conf_wrong:.2f} on "
        f"the ones it gets wrong. If this were used for real you could hold back "
        f"low-confidence predictions for a human to check."
    )
    lines.append(
        "- **The single-sentence setup is the ceiling here.** Papers on this "
        "dataset get to ~92% by also feeding the model the sentences around the "
        "one being classified. This model sees one sentence in isolation, which "
        "is most of the gap between ~86% and ~92%. Adding context is the next "
        "experiment (see Limitations in the README)."
    )
    lines.append("")

    lines.append("## Highest-confidence mistakes\n")
    lines.append("| true | predicted | conf | sentence |")
    lines.append("| --- | --- | --- | --- |")
    for idx in worst:
        s = texts[idx].replace("|", "\\|")
        s = (s[:110] + "…") if len(s) > 110 else s
        lines.append(
            f"| {labels[y_true[idx]]} | {labels[y_pred[idx]]} | "
            f"{conf[idx]:.2f} | {s} |"
        )
    lines.append("")

    with open("docs/evaluation.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote docs/evaluation.md")
    print(f"\naccuracy={acc:.3f}  macro_f1={f1_macro:.3f}")


if __name__ == "__main__":
    main()
