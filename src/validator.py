import os

import pandas as pd


class DataValidator:
    @staticmethod
    def validate(path: str) -> pd.DataFrame:
        """Validate the input CSV file and return a parsed DataFrame.

        Raises:
            FileNotFoundError: If the path does not exist on disk.
            ValueError: If the CSV cannot be parsed, the DataFrame has zero rows,
                        or the 'close' column is absent.
        Returns:
            pd.DataFrame: The parsed DataFrame on success.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"CSV parse error: {exc}") from exc

        if len(df) == 0:
            raise ValueError("Dataset is empty")

        if "close" not in df.columns:
            raise ValueError("Missing required column: close")

        return df
