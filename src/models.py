"""Forecasting models for day-ahead hourly electricity prices.

All models share the same interface: fit(X_train, y_train) / predict(X) on
the tabular feature frame from features.py, so they can be swapped into the
same walk-forward harness. The LSTM instead consumes a raw price-history
window (see SequenceLSTM) but is wrapped to expose the same predict(X)
signature by carrying its own lookback buffer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import torch
import torch.nn as nn


class NaivePersistenceModel:
    """Predicts price[t] = price[t-168] (same hour, same weekday, last week).

    Electricity prices have strong weekly seasonality (weekday/weekend,
    business-hour demand cycles repeat), so this is a genuinely strong,
    widely-used baseline in the forecasting literature, not a strawman.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaivePersistenceModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["price_lag_168h"].to_numpy()


class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class XGBoostModel:
    def __init__(self, n_estimators: int = 300, max_depth: int = 5, learning_rate: float = 0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    """Sequence model: predicts price[t] from the raw price sequence
    [t-168, ..., t-25] (i.e. the week of history available before the
    day-ahead cutoff, excluding the last 24h the forecast must not see).

    Does not use the tabular feature frame -- built directly from the price
    series passed to fit/predict via `price_series`, keyed by the same
    index as X, so it can slot into the same walk-forward loop.
    """

    def __init__(self, lookback: int = 144, hidden_size: int = 32, epochs: int = 15, lr: float = 1e-3):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.net = _LSTMNet(hidden_size=hidden_size)
        self.price_history: pd.Series | None = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def _make_sequences(self, target_index: pd.DatetimeIndex, price_history: pd.Series) -> np.ndarray:
        """Vectorized sequence extraction: reindexes price_history onto a
        complete hourly grid once, then slices by integer position for every
        target timestamp (no per-row pandas .loc calls, which is what made
        the naive version too slow to run at full scale).
        """
        full_range = pd.date_range(price_history.index.min(), price_history.index.max(), freq="h")
        hourly = price_history.reindex(full_range).ffill().bfill()
        values = hourly.to_numpy()
        origin = full_range[0]

        seqs = np.empty((len(target_index), self.lookback), dtype=np.float64)
        for row, t in enumerate(target_index):
            # Integer hour offset of t from the history's start -- computed
            # arithmetically so target timestamps beyond price_history's own
            # range (i.e. the test period itself) resolve correctly; only
            # the window [t-25-lookback+1, t-25] is ever read, which stays
            # inside price_history since price_history excludes the test
            # period being predicted.
            t_pos = int((t - origin) / pd.Timedelta(hours=1))
            end_pos = t_pos - 25  # t-25h: last hour visible before the day-ahead cutoff
            start_pos = end_pos - self.lookback + 1
            if start_pos < 0:
                # Not enough history this far back (only happens for the very
                # first rows of the whole dataset) -- pad by repeating the
                # earliest known value rather than fabricating a trend.
                pad = np.full(-start_pos, values[0])
                seqs[row] = np.concatenate([pad, values[0:end_pos + 1]])
            else:
                seqs[row] = values[start_pos:end_pos + 1]
        return seqs

    def fit(self, X: pd.DataFrame, y: pd.Series, price_history: pd.Series | None = None,
            batch_size: int = 256, verbose: bool = False) -> "LSTMModel":
        self.price_history = price_history if price_history is not None else y
        seqs = self._make_sequences(X.index, self.price_history)
        self.mean_, self.std_ = seqs.mean(), seqs.std() + 1e-8
        seqs_norm = (seqs - self.mean_) / self.std_
        y_norm = (y.to_numpy() - self.mean_) / self.std_

        X_t = torch.tensor(seqs_norm, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        n = X_t.shape[0]

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                opt.zero_grad()
                pred = self.net(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)
            if verbose:
                print(f"  epoch {epoch + 1}/{self.epochs} loss={epoch_loss / n:.4f}")
        return self

    def predict(self, X: pd.DataFrame, price_history: pd.Series | None = None) -> np.ndarray:
        history = price_history if price_history is not None else self.price_history
        seqs = self._make_sequences(X.index, history)
        seqs_norm = (seqs - self.mean_) / self.std_
        X_t = torch.tensor(seqs_norm, dtype=torch.float32).unsqueeze(-1)
        self.net.eval()
        with torch.no_grad():
            pred_norm = self.net(X_t).numpy()
        return pred_norm * self.std_ + self.mean_
