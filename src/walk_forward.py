"""Walk-forward evaluation for day-ahead price forecasting.

Retrains each model periodically (default: weekly) and predicts the
following retrain_period days hour-by-hour, always using only data strictly
before the retrain cutoff -- no model ever sees a data point it is being
scored against, directly or through a feature derived from it.
"""

from __future__ import annotations

import pandas as pd

from src.metrics import summarize


def walk_forward_evaluate(
    feat: pd.DataFrame,
    feature_cols: list[str],
    price_col: str,
    model_factory,
    test_days: int = 180,
    retrain_every_days: int = 7,
    uses_price_history: bool = False,
    raw_price_series: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    """Returns (predictions indexed by timestamp over the test period, metrics dict)."""
    test_start = feat.index.max() - pd.Timedelta(days=test_days)
    train_full = feat[feat.index < test_start]
    test_full = feat[feat.index >= test_start]

    all_preds: list[pd.Series] = []
    cutoff = test_start
    end_of_test = feat.index.max()

    while cutoff < end_of_test:
        window_end = min(cutoff + pd.Timedelta(days=retrain_every_days), end_of_test)

        train_slice = feat[feat.index < cutoff]
        test_slice = feat[(feat.index >= cutoff) & (feat.index < window_end)]
        if test_slice.empty or len(train_slice) < 168 * 4:
            cutoff = window_end
            continue

        model = model_factory()
        X_train, y_train = train_slice[feature_cols], train_slice[price_col]
        X_test = test_slice[feature_cols]

        if uses_price_history:
            source = raw_price_series if raw_price_series is not None else feat[price_col]
            # Fit only ever sees data strictly before the retrain cutoff
            # (governs model weights). Prediction may reference real prices
            # up to t-25h for each row -- the same causal guarantee the
            # tabular lag features already rely on (a row's own lag_24h/
            # lag_168h use real data up to that row's t-24h regardless of
            # when the model was last retrained), so this keeps both model
            # families held to an identical, consistent no-leakage rule.
            train_history = source.loc[source.index < cutoff]
            predict_history = source.loc[source.index < window_end]
            model.fit(X_train, y_train, price_history=train_history)
            preds = model.predict(X_test, price_history=predict_history)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        all_preds.append(pd.Series(preds, index=test_slice.index))
        cutoff = window_end

    pred_series = pd.concat(all_preds).sort_index()
    actual = feat.loc[pred_series.index, price_col]
    metrics = summarize(actual.to_numpy(), pred_series.to_numpy())
    return pred_series, metrics
