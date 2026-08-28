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

Runs on a free-tier **AWS EC2** instance (`t3.micro`, Amazon Linux 2023),
provisioned with **Terraform** ([`infra/`](infra/)). **GitHub Actions**
([`.github/workflows/ci-deploy.yml`](.github/workflows/ci-deploy.yml)) runs the
test suite, builds a CPU-only image, pushes it to **GHCR**, then SSHes to the
instance to pull and restart the container.

The image is code-only; the model checkpoint is mounted as a read-only volume
so images stay small and the model has its own lifecycle.

```bash
# Local
docker build -t clinical-text-classifier .
docker run -p 8000:8000 \
  -v "$(pwd)/model_output/final:/app/model_output/final:ro" \
  clinical-text-classifier
```

**Live:** http://107.21.9.181 &nbsp;·&nbsp; [`/docs`](http://107.21.9.181/docs) &nbsp;·&nbsp; [`/health`](http://107.21.9.181/health)

```bash
curl -X POST http://107.21.9.181/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Patients were randomly assigned to prednisolone or placebo ."}'
# -> {"label":"methods","confidence":0.99,...}
```

## Status

- [x] Data loading interface + PubMed 20k RCT loader
- [x] Training script with MLflow tracking
- [x] FastAPI serving layer
- [x] Test suite
- [x] Dockerfile
- [x] AWS deployment (Terraform + EC2)
- [x] CI/CD (GitHub Actions -> GHCR -> EC2)
- [ ] MIMIC-III loader (pending PhysioNet credentialing)
