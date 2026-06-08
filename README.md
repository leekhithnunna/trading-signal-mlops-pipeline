# MLOps Batch Job — Task 0 Technical Assessment

A production-quality MLOps batch job that processes financial OHLCV time-series data to compute a rolling-mean crossover signal. Demonstrates three core MLOps pillars: **reproducibility**, **observability**, and **deployment readiness**.

---

## Project Structure

```
.
├── run.py                    # Batch job entry point (CLI orchestrator)
├── config.yaml               # Default configuration (seed, window, version)
├── data.csv                  # Input OHLCV dataset (10,000 rows, BTC/USD 1-min)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image definition (python:3.9-slim)
├── src/
│   ├── config.py             # Config dataclass + ConfigLoader
│   ├── validator.py          # DataValidator (CSV validation)
│   ├── processor.py          # SignalProcessor (rolling mean + signal)
│   ├── metrics.py            # MetricsWriter (JSON output)
│   └── logger.py             # setup_logger (UTC timestamps, file + stdout)
└── tests/
    ├── test_config_loader.py  # Unit tests: ConfigLoader
    ├── test_validator.py      # Unit tests: DataValidator
    ├── test_processor.py      # Unit tests: SignalProcessor
    ├── test_pipeline.py       # End-to-end pipeline tests
    └── test_properties.py     # Property-based tests (Hypothesis)
```

---

## Pipeline Overview

```
CLI args → Logger → ConfigLoader → numpy.random.seed()
       → DataValidator → SignalProcessor (rolling_mean, signal)
       → MetricsWriter → metrics.json
```

On any failure, an error `metrics.json` is always written before exit.

---

## Local Setup

### Prerequisites

- Python 3.9+ (tested on 3.12)
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

### Arguments

| Argument     | Description                        |
|--------------|------------------------------------|
| `--input`    | Path to input CSV file             |
| `--config`   | Path to YAML configuration file    |
| `--output`   | Path for JSON metrics output       |
| `--log-file` | Path for structured log output     |

### config.yaml format

```yaml
seed: 42       # Integer — numpy random seed for reproducibility
window: 5      # Integer — rolling mean window size
version: "v1"  # String  — job version tag
```

---

## Running Tests

```bash
# All unit + property-based tests
pytest tests/ -v

# Expected: 30 passed
```

---

## Example metrics.json (success)

```json
{
    "version": "v1",
    "rows_processed": 10000,
    "metric": "signal_rate",
    "value": 0.4989,
    "latency_ms": 78,
    "seed": 42,
    "status": "success"
}
```

## Example metrics.json (error)

```json
{
    "version": "v1",
    "status": "error",
    "error_message": "Input file not found: missing.csv"
}
```

---

## Docker

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container (uses bundled data.csv + config.yaml)

```bash
docker run --rm mlops-task
```

Expected output: structured log lines to stdout, followed by the JSON metrics payload. Exit code 0 on success.

### Run with custom files

```bash
docker run --rm \
  -v $(pwd)/my_data.csv:/app/data.csv \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  mlops-task
```

### Verify failure exit code

```bash
docker run --rm mlops-task sh -c \
  "python run.py --input missing.csv --config /app/config.yaml --output /app/metrics.json --log-file /app/run.log"
echo "Exit: $?"   # Should be 1
```

---

## Output Files

| File           | Description                                          |
|----------------|------------------------------------------------------|
| `metrics.json` | Job metrics — always written (success or error)      |
| `run.log`      | Structured log with UTC timestamps                   |

---

## Design Notes

- **Reproducibility**: `numpy.random.seed(config.seed)` is called before any processing. Two identical runs always produce identical `value`/`signal_rate`.
- **Observability**: Every pipeline stage emits an INFO log; all exceptions emit ERROR with full stack trace.
- **Error resilience**: `metrics.json` is written on both success and failure paths — monitoring systems never see a missing output file.
- **No hardcoded paths**: all file paths are provided via CLI arguments.
