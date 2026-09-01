# Experiments

Three models, same PubMed 20k RCT test split (29,578 held-out sentences), same
evaluation script. Each row is one deliberate step up.

| model | input | accuracy | macro F1 | detail |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | one sentence, bag of 1–2 grams | 0.822 | 0.753 | [baseline.md](baseline.md) |
| DistilBERT | one sentence | 0.866 | 0.806 | [evaluation.md](evaluation.md) |
| DistilBERT | sentence + its two neighbours | **0.911** | **0.858** | [evaluation_context.md](evaluation_context.md) |

## 1. Bag of words → 82%

A linear model on TF-IDF unigrams + bigrams. Gets to 82% on its own, which says
most of this task is lexical — numbers and "p <" point at `results`, "we
conclude" / "these findings" point at `conclusions`. ~20 seconds to fit.

## 2. Transformer, single sentence → 87%

`distilbert-base-uncased`, fully fine-tuned, one sentence at a time. +4.4 points
over the baseline — the part that needs word order and phrasing rather than
word counts. This is the deployed model.

## 3. Transformer + context → 91%

Same model, but the input becomes

```
[previous sentence] [SEP] [target sentence] [SEP] [next sentence]
```

with `max_length` bumped to 256. The label is still the *target* sentence's
role; the neighbours are only there for context. Empty string at abstract
boundaries.

+4.5 points over single-sentence. **Why it helps:** a `conclusions` sentence
and a `results` sentence look nearly identical in isolation ("X improved
significantly, p < 0.05"). What separates them is position — end of the
abstract vs the middle. The neighbouring sentences carry that signal.
`conclusions` F1 goes 0.83 → 0.95 and `results` 0.92 → 0.95.

Still hard after context: `objective` (F1 0.68). It's genuinely close to
`background`, and the neighbouring sentence often doesn't settle it.

## What I skipped and why

- **Hyperparameter sweep.** v1's per-epoch numbers showed the model converges
  in about one epoch and `2e-5` is fine; a sweep wasn't going to move the
  result enough to be worth 5+ hours of training.
- **Deploying v2.** The `/predict` API takes a single sentence. Serving the
  context model would mean changing that contract to accept the neighbours.
  v1 stays deployed; v2 is a documented result.

## Reproduce

```bash
# baseline
python -m scripts.baseline

# train the context model (~3 h on an M-series GPU)
python -m src.train --dataset pubmed_20k_rct_context \
  --max_length 256 --epochs 3 --output_dir ./model_output/context

# evaluate either model
python -m scripts.evaluate
python -m scripts.evaluate \
  --model-dir model_output/context/final \
  --dataset pubmed_20k_rct_context --max-length 256 \
  --out docs/evaluation_context.md --cm-out docs/confusion_matrix_context.png \
  --title "with ±1 sentence context"
```

All three runs are logged in MLflow under the `clinical-text-classifier`
experiment.
