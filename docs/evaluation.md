# Evaluation

Model: `distilbert-base-uncased`, fully fine-tuned for 3 epochs. Evaluated on the full PubMed 20k RCT **test** split (29,578 sentences the model never saw during training).

The task: given one sentence from a medical-paper abstract, predict its role — `background`, `objective`, `methods`, `results`, or `conclusions`.

This file is generated. Re-run it with `python -m scripts.evaluate`.

## Headline numbers

| metric | value |
| --- | --- |
| accuracy | 0.866 |
| macro F1 | 0.806 |
| mean confidence when right | 0.950 |
| mean confidence when wrong | 0.792 |

## Per-class

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| background | 0.642 | 0.767 | 0.699 | 3,077 |
| conclusions | 0.852 | 0.817 | 0.834 | 4,571 |
| methods | 0.925 | 0.947 | 0.936 | 9,884 |
| objective | 0.745 | 0.573 | 0.648 | 2,333 |
| results | 0.920 | 0.910 | 0.915 | 9,713 |

## Confusion matrix

![confusion matrix](confusion_matrix.png)

## What I take from this

- **`methods` and `results` are easy, `objective` and `background` are hard.** The easy ones have giveaway wording — numbers, p-values, "we measured", "in conclusion". The hard ones are both short, general statements at the very start of an abstract, and telling "here is the problem" from "here is what we tested" is genuinely ambiguous.
- **Most common mistake: `objective` predicted as `background`** (848 sentences), which lines up with the point above.
- **The model knows when it's unsure.** Average confidence is 0.95 on the ones it gets right vs 0.79 on the ones it gets wrong. If this were used for real you could hold back low-confidence predictions for a human to check.
- **The single-sentence setup is the ceiling here.** Papers on this dataset get to ~92% by also feeding the model the sentences around the one being classified. This model sees one sentence in isolation, which is most of the gap between ~86% and ~92%. Adding context is the next experiment (see Limitations in the README).

## Highest-confidence mistakes

| true | predicted | conf | sentence |
| --- | --- | --- | --- |
| methods | results | 1.00 | ( @ ) In the MIA group and ADA group , after surgery , the increased dosage of fentanyl citrate was less than … |
| background | results | 1.00 | However , the subgroup of subjects , which inhaled UFP during the first exposure , exhibited a significant inc… |
| conclusions | results | 1.00 | Rifampicin significantly increased the mean area under the plasma concentration-time curve ( AUC ) of ( R ) - … |
| methods | results | 1.00 | No pain was reported in the saffron group , whereas the indomethacin group experienced pain before @ hours ( P… |
| objective | conclusions | 1.00 | Once-daily losartan reduces BP in a dose-dependent manner and is well tolerated in hypertensive children aged … |
| conclusions | results | 1.00 | PDT was associated with a significant decrease in bleeding scores ( P = @ ) as well as inflammatory exudation … |
