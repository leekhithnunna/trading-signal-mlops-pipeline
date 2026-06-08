import json
import time

import pandas as pd

from src.config import Config


class MetricsWriter:
    @staticmethod
    def write_success(path: str, df: pd.DataFrame, config: Config, start_time: float) -> None:
        """Write a success metrics JSON file.

        Computes rows_processed, signal_rate, and latency_ms from the
        processed DataFrame and config, then writes the result to path
        with 4-space indentation.
        """
        rows_processed = len(df)
        signal_rate = round(df["signal"].mean(), 4)
        latency_ms = int((time.time() - start_time) * 1000)

        payload = {
            "version": config.version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": config.seed,
            "status": "success",
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)

    @staticmethod
    def write_error(path: str, version: str, error_message: str) -> None:
        """Write an error metrics JSON file.

        Writes a minimal error payload to path with 4-space indentation.
        Falls back to 'v1' if version is empty or None.
        """
        payload = {
            "version": version if version else "v1",
            "status": "error",
            "error_message": error_message,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
