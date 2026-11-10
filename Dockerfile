# Inference service. Build after `make train` has written artifacts/models/lstm_best.pt:
#   docker build -t biotoolkit . && docker run -p 8000:8000 biotoolkit
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY biotoolkit ./biotoolkit
RUN pip install --no-cache-dir .
COPY artifacts/models/lstm_best.pt ./artifacts/models/lstm_best.pt

ENV MODEL_PATH=/app/artifacts/models/lstm_best.pt
EXPOSE 8000
CMD ["uvicorn", "biotoolkit.serve:app", "--host", "0.0.0.0", "--port", "8000"]
