# Serves the trained model via FastAPI. Assumes model_output/final exists
# (train first, or mount/copy a trained checkpoint in).
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch. The default wheels pull a multi-GB CUDA stack
# (cudnn, nccl, triton, ...) that is dead weight in a CPU serving container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Serving deps only (see requirements-api.txt vs the full requirements.txt).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY model_output/final ./model_output/final

ENV MODEL_DIR=/app/model_output/final

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
