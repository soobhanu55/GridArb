"""SHAP explainability for the tree-based forecasters (XGBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_values(tree_model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (one row per sample, one column per feature) for a fitted sklearn/XGBoost tree model."""
    return shap.TreeExplainer(tree_model).shap_values(X)


def global_importance(values: np.ndarray, columns: list[str]) -> dict:
    """Mean |SHAP| per feature, largest first: how much each feature moves the forecast, in EUR/MWh."""
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return {"features": [columns[i] for i in order], "mean_abs_shap": [float(mean_abs[i]) for i in order]}


def save_summary_plot(values: np.ndarray, X: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(values, X, show=False, max_display=12)
    plt.title("What drives the XGBoost price forecast (SHAP, EUR/MWh)")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
