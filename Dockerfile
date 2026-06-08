# MLOps Batch Job — Docker image
# Base: python:3.9-slim (Req 10.1)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching (Req 10.2)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code (Req 10.2)
COPY run.py .
COPY src/ ./src/

# Copy bundled data and config (Req 10.3)
COPY data.csv .
COPY config.yaml .

# Default CMD:
#   1. Run the pipeline with default paths (Req 10.4)
#   2. On success: print metrics.json to stdout and exit 0 (Req 10.5)
#   3. On failure: exit non-zero (Req 10.6)
CMD ["sh", "-c", \
     "python run.py \
        --input /app/data.csv \
        --config /app/config.yaml \
        --output /app/metrics.json \
        --log-file /app/run.log \
     && cat /app/metrics.json"]
