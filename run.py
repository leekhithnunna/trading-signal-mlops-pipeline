"""BatchJob orchestrator: wires all pipeline components together."""

import argparse
import datetime
import sys
import time
import traceback

import numpy

from src.config import ConfigLoader
from src.logger import setup_logger
from src.metrics import MetricsWriter
from src.processor import SignalProcessor
from src.validator import DataValidator


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments.

    All four arguments are required. Missing arguments cause argparse to
    print a usage message and exit with a non-zero status automatically.
    """
    parser = argparse.ArgumentParser(
        description="MLOps batch job: compute rolling-mean crossover signal."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the JSON metrics output file.",
    )
    parser.add_argument(
        "--log-file",
        required=True,
        dest="log_file",
        help="Path for the log output file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full batch-job pipeline.

    Pipeline order:
        1. Parse CLI args
        2. Initialise Logger
        3. Start wall-clock timer
        4. Load & validate Config  → seed numpy RNG
        5. Load & validate Dataset
        6. Compute rolling_mean + signal  (SignalProcessor)
        7. Compute & write metrics        (MetricsWriter → success JSON)
        8. Exit 0

    On any exception:
        - Log at ERROR level with full stack trace
        - Write error metrics JSON
        - Exit non-zero
    """
    args = parse_args()
    logger = setup_logger(args.log_file)

    # Wall-clock timer starts immediately after the logger is ready so that
    # total job latency (including config and data loading) is captured.
    start_time = time.time()

    # Req 7.2: INFO log on job start (including UTC timestamp)
    utc_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Job started at %s", utc_now)

    # config_version is used in the except block; default to 'v1' in case the
    # exception occurs before config is loaded (Req 8.4 / design note).
    config_version: str = "v1"

    try:
        # --- Config ----------------------------------------------------------
        config = ConfigLoader.load(args.config)

        # Req 6.4: INFO log when config is loaded
        logger.info("Config loaded from %s", args.config)

        # Req 6.5: INFO log when config is validated
        logger.info(
            "Config validated — seed=%d, window=%d, version=%s",
            config.seed,
            config.window,
            config.version,
        )

        # Capture the real version for use in any subsequent error path
        config_version = config.version

        # Req 2.5 / 9.2: Seed numpy RNG for reproducibility
        numpy.random.seed(config.seed)

        # --- Data ------------------------------------------------------------
        df = DataValidator.validate(args.input)

        # Req 6.6: INFO log when dataset is loaded
        logger.info(
            "Dataset loaded — %d rows from %s", len(df), args.input
        )

        # --- Signal processing -----------------------------------------------
        # Req 6.7: INFO log when rolling mean computation begins
        logger.info("Computing rolling mean (window=%d)", config.window)

        # Req 6.8: INFO log when signal generation begins
        logger.info("Generating binary signal")

        df = SignalProcessor.process(df, config.window)

        # --- Metrics ---------------------------------------------------------
        # Req 6.9: INFO log when metrics are computed
        rows_processed = len(df)
        signal_rate = round(df["signal"].mean(), 4)
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Metrics computed — rows_processed=%d, signal_rate=%s, latency_ms=%d",
            rows_processed,
            signal_rate,
            latency_ms,
        )

        MetricsWriter.write_success(args.output, df, config, start_time)

        # Req 6.10: INFO log on successful job completion
        logger.info("Job completed successfully")

        sys.exit(0)

    except Exception as exc:  # noqa: BLE001
        # Req 8.4 / 6.11: ERROR log with full stack trace
        logger.error(str(exc), exc_info=True)

        # Req 8.2: always write error metrics before exiting
        MetricsWriter.write_error(args.output, config_version, str(exc))

        # Req 8.3: exit non-zero on failure
        sys.exit(1)


if __name__ == "__main__":
    main()
