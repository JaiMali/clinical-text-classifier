# Serves the trained model via FastAPI.
#
# The model checkpoint is NOT baked into the image -- it is mounted at run time
# so images stay code-only (faster builds/pushes) and the model has its own
# lifecycle. Provide it as a read-only volume at $MODEL_DIR:
#
#   docker run -p 8000:8000 -v /path/to/model_output/final:/app/model_output/final:ro \
#     clinical-text-classifier
#
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch. The default wheels pull a multi-GB CUDA stack
# (cudnn, nccl, triton, ...) that is dead weight in a CPU serving container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Serving deps only (see requirements-api.txt vs the full requirements.txt).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

ENV MODEL_DIR=/app/model_output/final

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
