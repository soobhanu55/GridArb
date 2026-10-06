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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if present, CPU in CI/tests


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


class RandomForestModel:
    """Bagged-trees baseline (absorbed from the DemandLens demand-forecasting repo)."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 12):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
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


class _MLPNet(nn.Module):
    def __init__(self, lookback: int, hidden_size: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(lookback, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, 1)
        return self.body(x.squeeze(-1)).squeeze(-1)


class _CNNNet(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, hidden_size, kernel_size=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class _CNNLSTMNet(nn.Module):
    """Conv layer extracts local patterns, LSTM models their order over the window."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(feats)
        return self.head(out[:, -1, :]).squeeze(-1)


class SequenceModel:
    """Base for neural models that read the raw price window instead of the
    tabular feature frame: predicts price[t] from the sequence
    [t-168, ..., t-25] (i.e. the week of history available before the
    [t-168, ..., t-25] (i.e. the week of history available before the
    day-ahead cutoff, excluding the last 24h the forecast must not see).
    Subclasses only choose the network (_build_net); windowing, scaling and
    training are shared, so every architecture is held to the same no-leakage rule.

    Does not use the tabular feature frame -- built directly from the price
    series passed to fit/predict via `price_series`, keyed by the same
    index as X, so it can slot into the same walk-forward loop.
    """

    def __init__(self, lookback: int = 144, hidden_size: int = 32, epochs: int = 15, lr: float = 1e-3,
                 seed: int = 42):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        torch.manual_seed(seed)  # reproducible weights + batch order
        self.net = self._build_net().to(DEVICE)
        self.price_history: pd.Series | None = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

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
                head = values[0:max(end_pos + 1, 0)]  # end_pos < 0: the whole window precedes the history
                seqs[row] = np.concatenate([np.full(self.lookback - len(head), values[0]), head])
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

        X_t = torch.tensor(seqs_norm, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_t = torch.tensor(y_norm, dtype=torch.float32).to(DEVICE)
        n = X_t.shape[0]

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
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
        X_t = torch.tensor(seqs_norm, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        self.net.eval()
        with torch.no_grad():
            pred_norm = self.net(X_t).cpu().numpy()
        return pred_norm * self.std_ + self.mean_


class LSTMModel(SequenceModel):
    def _build_net(self) -> nn.Module:
        return _LSTMNet(hidden_size=self.hidden_size)


class MLPModel(SequenceModel):
    def _build_net(self) -> nn.Module:
        return _MLPNet(self.lookback, hidden_size=2 * self.hidden_size)


class CNNModel(SequenceModel):
    def _build_net(self) -> nn.Module:
        return _CNNNet(hidden_size=self.hidden_size)


class CNNLSTMModel(SequenceModel):
    def _build_net(self) -> nn.Module:
        return _CNNLSTMNet(hidden_size=self.hidden_size)


class _ClassicalForecaster:
    """Univariate time-series models (absorbed from DemandLens: Holt-Winters, ARIMA family).

    They ignore the tabular features: fit() keeps the tail of the training prices,
    predict() forecasts forward from the end of training to each requested timestamp.
    In the walk-forward harness the test window directly follows the training data,
    so the horizon is at most `retrain_every_days` * 24 steps.
    """

    history_hours = 56 * 24  # recent history is enough, and keeps fitting fast

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ClassicalForecaster":
        y = y.sort_index()
        # fill any missing hours so the series has a regular hourly frequency
        y = y.asfreq("h").interpolate(limit_direction="both").iloc[-self.history_hours:]
        self.last_ts_ = y.index[-1]
        self.fitted_ = self._fit(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = ((X.index - self.last_ts_) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        forecast = np.asarray(self.fitted_.forecast(int(steps.max())))
        return forecast[steps - 1]

    def _fit(self, y: pd.Series):
        raise NotImplementedError


class HoltWintersModel(_ClassicalForecaster):
    """Additive Holt-Winters exponential smoothing with a weekly (168h) season."""

    def _fit(self, y: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        return ExponentialSmoothing(
            y.to_numpy(), trend=None, seasonal="add", seasonal_periods=168,
            initialization_method="estimated",
        ).fit()


class SARIMAModel(_ClassicalForecaster):
    """SARIMA(1,0,1)x(1,1,0,24): short-memory ARMA terms plus a daily seasonal difference."""

    history_hours = 28 * 24

    def _fit(self, y: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(y.to_numpy(), order=(1, 0, 1), seasonal_order=(1, 1, 0, 24)).fit(disp=False, maxiter=50)
