import pandas as pd


class SignalProcessor:
    @staticmethod
    def process(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Compute the rolling mean and binary signal for a DataFrame.

        Adds two columns to the DataFrame (mutating it in place):
        - ``rolling_mean``: the rolling mean of ``close`` over ``window`` periods.
          Leading ``NaN`` values (first ``window - 1`` rows) are preserved.
        - ``signal``: 1 when ``close > rolling_mean``, 0 otherwise (including rows
          where ``rolling_mean`` is NaN, because NaN comparisons evaluate to False).

        Args:
            df: A validated DataFrame containing at least a ``close`` column.
            window: The rolling-window size sourced from Config.

        Returns:
            The mutated DataFrame with ``rolling_mean`` and ``signal`` columns added.
        """
        df["rolling_mean"] = df["close"].rolling(window=window).mean()
        df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)
        return df
