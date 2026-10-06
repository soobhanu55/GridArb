import numpy as np
import pandas as pd

from src.explain import global_importance, shap_values
from src.models import XGBoostModel


def test_shap_ranks_the_feature_that_actually_drives_the_target():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=500), "noise": rng.normal(size=500)})
    y = 10 * X["signal"]
    model = XGBoostModel(n_estimators=60).fit(X, y)
    imp = global_importance(shap_values(model.model, X), list(X.columns))
    assert imp["features"][0] == "signal"
    assert imp["mean_abs_shap"][0] > 5 * imp["mean_abs_shap"][1]


def test_shap_values_add_up_to_the_prediction():
    """Core SHAP property: base value + sum of per-feature contributions == model output."""
    import shap

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = 2 * X["a"] - X["b"]
    model = XGBoostModel(n_estimators=40).fit(X, y)
    explainer = shap.TreeExplainer(model.model)
    recon = explainer.expected_value + explainer.shap_values(X).sum(axis=1)
    np.testing.assert_allclose(recon, model.predict(X), atol=0.02)  # XGBoost predicts in float32
