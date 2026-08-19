import numpy as np
import pandas as pd
import pytest

from src.models import NaivePersistenceModel, LSTMModel


def test_naive_persistence_returns_lag_168_column_unchanged():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    X = pd.DataFrame({"price_lag_168h": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)
    model = NaivePersistenceModel().fit(X, pd.Series([0] * 5))
    preds = model.predict(X)
    assert preds.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_lstm_make_sequences_hand_calculated():
    """price[i] = i. For target timestamp at position t_pos, the window
    should be exactly [t_pos-25-lookback+1 .. t_pos-25] (inclusive), i.e.
    lookback consecutive integers ending 25 hours before the target.
    """
    n_hours = 300
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    price_history = pd.Series(np.arange(n_hours, dtype=float), index=idx)

    model = LSTMModel(lookback=10)
    target_index = idx[[200, 250]]
    seqs = model._make_sequences(target_index, price_history)

    assert seqs.shape == (2, 10)
    # target at position 200: window ends at 200-25=175, starts at 175-9=166
    assert seqs[0].tolist() == list(range(166, 176))
    # target at position 250: window ends at 225, starts at 216
    assert seqs[1].tolist() == list(range(216, 226))


def test_lstm_make_sequences_pads_when_not_enough_history():
    """A target very close to the start of price_history's range doesn't
    have lookback+25 hours of history behind it -- the earliest known value
    should be repeated to pad, not crash or silently truncate.
    """
    n_hours = 50
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    price_history = pd.Series(np.arange(n_hours, dtype=float), index=idx)

    model = LSTMModel(lookback=20)
    target_index = idx[[26]]  # end_pos = 26-25=1, start_pos = 1-19=-18 -> needs padding
    seqs = model._make_sequences(target_index, price_history)

    assert seqs.shape == (1, 20)
    # first 18 entries padded with price_history[0] == 0.0, then 0, 1 (positions 0 and 1)
    assert seqs[0][:18].tolist() == [0.0] * 18
    assert seqs[0][18:].tolist() == [0.0, 1.0]


def test_lstm_fit_predict_smoke_test_no_nans():
    """Full fit/predict round trip on a small synthetic set shouldn't
    produce NaNs (regression check for the earlier truncated-history bug).
    """
    n_hours = 24 * 15
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    price = pd.Series(50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24), index=idx)

    train_idx = idx[168:n_hours - 24]
    test_idx = idx[n_hours - 24:]

    model = LSTMModel(lookback=48, epochs=2)
    model.fit(pd.DataFrame(index=train_idx), price.loc[train_idx], price_history=price.loc[idx < idx[n_hours - 24]])
    preds = model.predict(pd.DataFrame(index=test_idx), price_history=price)

    assert not np.isnan(preds).any()
    assert len(preds) == 24
