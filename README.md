# Clinical Text Classifier

Fine-tunes DistilBERT to classify clinical text sentences, served as a REST API and deployed via Docker.

Built as a resume/portfolio project demonstrating the ML engineering pipeline end-to-end: training, experiment tracking, serving, testing, and deployment.

## Architecture

```
data/loader.py      -- dataset-agnostic loading interface (swap datasets without touching anything else)
train.py             -- fine-tunes DistilBERT, tracks every run in MLflow
api/main.py          -- FastAPI service wrapping the trained model
tests/                -- pytest suite for the data contract and the API
Dockerfile            -- containerizes the serving layer for deployment
```

### Why an isolated data-loading layer

This project currently trains on **PubMed 20k RCT** (public, no credentialing required). A planned v2 will retrain on **MIMIC-III** once PhysioNet access is approved. Every dataset-specific detail (source, column names, label taxonomy) lives entirely in `data/loader.py` behind a single `load_dataset(name)` function returning a standardized `DatasetBundle`. Training, serving, and tests never import a dataset directly -- so swapping datasets is a new loader function, not a rewrite.

### Why DistilBERT (not QLoRA / a large LLM)

DistilBERT (66M params) is small enough to fully fine-tune directly. QLoRA -- quantized low-rank adaptation -- solves a different problem: making fine-tuning feasible for much larger models (billions of parameters) that can't fit in GPU memory for full fine-tuning. It isn't needed here, but is used in a separate project fine-tuning an 8B-parameter open-weight LLM.

## Running locally

```bash
pip install -r requirements.txt

# Train (logs to MLflow; view with `mlflow ui`)
python -m src.train --dataset pubmed_20k_rct --epochs 3

# Serve
uvicorn src.api.main:app --reload

# Test
pytest                      # unit tests only
pytest -m integration       # includes real dataset download
```

## Deployment

```bash
docker build -t clinical-text-classifier .
docker run -p 8000:8000 clinical-text-classifier
```

Deployed on AWS EC2 at: _(add URL once live)_

## Status

- [x] Data loading interface + PubMed 20k RCT loader
- [x] Training script with MLflow tracking
- [x] FastAPI serving layer
- [x] Test suite
- [x] Dockerfile
- [ ] AWS deployment
- [ ] CI/CD (GitHub Actions)
- [ ] MIMIC-III loader (pending PhysioNet credentialing)
