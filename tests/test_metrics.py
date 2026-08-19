import numpy as np
import pytest

from src.metrics import mae, r_squared, rmse, summarize


def test_mae_hand_calculated():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |2| + |2| + |3| = 7, /3 = 2.3333...
    assert mae(y_true, y_pred) == pytest.approx(7 / 3, abs=1e-9)


def test_rmse_hand_calculated():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt((9+16)/2) = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), abs=1e-9)


def test_r_squared_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y_true, y_true.copy()) == pytest.approx(1.0, abs=1e-9)


def test_r_squared_predicting_the_mean_gives_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full(4, y_true.mean())
    assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)


def test_r_squared_hand_calculated_negative_case():
    """A prediction worse than always guessing the mean should score < 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 1.0, 5.0])  # deliberately anti-correlated-ish
    ss_res = (1 - 3) ** 2 + (2 - 1) ** 2 + (3 - 5) ** 2  # 4+1+4=9
    mean = 2.0
    ss_tot = (1 - mean) ** 2 + (2 - mean) ** 2 + (3 - mean) ** 2  # 1+0+1=2
    expected = 1 - ss_res / ss_tot  # 1 - 4.5 = -3.5
    assert r_squared(y_true, y_pred) == pytest.approx(expected, abs=1e-9)


def test_summarize_returns_all_three_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = summarize(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "r2"}
    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result["r2"] == pytest.approx(1.0, abs=1e-9)
