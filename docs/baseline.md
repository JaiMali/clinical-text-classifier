# Baseline: TF-IDF + logistic regression

A deliberately non-neural model on the same train/test split, so the DistilBERT numbers have a floor to be compared against. Bag of unigrams + bigrams, TF-IDF weighted, into a linear classifier.

This file is generated. Re-run with `python -m scripts.baseline`.

## Test set

| model | accuracy | macro F1 |
| --- | --- | --- |
| TF-IDF + logistic regression | 0.822 | 0.753 |
| DistilBERT (single sentence) | 0.866 | 0.806 |
| gain from the transformer | +0.044 | +0.053 |

## Per-class (baseline)

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| background | 0.629 | 0.628 | 0.628 | 3,077 |
| conclusions | 0.760 | 0.746 | 0.753 | 4,571 |
| methods | 0.872 | 0.929 | 0.900 | 9,884 |
| objective | 0.713 | 0.528 | 0.607 | 2,333 |
| results | 0.876 | 0.881 | 0.878 | 9,713 |

## What this tells me

- A bag of words already gets ~82%. Most of this task is just which words show up — numbers and "p <" point at `results`, "we conclude" / "these findings" point at `conclusions`.
- DistilBERT adds ~4% accuracy on top. That's the part that needs word order and phrasing rather than word counts.
- Both models are weakest on the same classes (`objective`, `background`), so that gap is about the task being genuinely ambiguous, not about the model.
