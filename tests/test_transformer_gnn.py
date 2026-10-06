"""The from-scratch Transformer and GCN, checked against PyTorch's reference implementations and for leakage."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src import zone_gnn
from src.transformer import (MultiHeadSelfAttention, PatchTransformerNet, TransformerModel,
                             scaled_dot_product_attention, sinusoidal_positions)


# --- Transformer -----------------------------------------------------------------------------------------

def test_attention_matches_torch_reference_and_rows_sum_to_one():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 10, 8) for _ in range(3))
    out, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 10), atol=1e-5)


def test_multihead_attention_matches_nn_multiheadattention_with_copied_weights():
    torch.manual_seed(1)
    ours = MultiHeadSelfAttention(32, 4)
    ref = nn.MultiheadAttention(32, 4, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.out.weight)
        ref.out_proj.bias.copy_(ours.out.bias)
    x = torch.randn(3, 12, 32)
    assert torch.allclose(ours(x), ref(x, x, x, need_weights=False)[0], atol=1e-5)


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(30, 4)


def test_sinusoidal_positions_start_with_sin0_cos0_and_are_distinct():
    pe = sinusoidal_positions(24, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    assert len({tuple(row.tolist()) for row in pe}) == 24


def test_patch_transformer_shapes_and_validation():
    net = PatchTransformerNet(lookback=144, patch=6, d_model=32, n_heads=4, n_layers=2)
    assert net(torch.randn(5, 144, 1)).shape == (5,)
    with pytest.raises(ValueError):
        PatchTransformerNet(lookback=100, patch=6)


def test_transformer_model_is_deterministic_and_ignores_the_future():
    idx = pd.date_range("2025-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(0)
    price = pd.Series(60 + 20 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 2, len(idx)), index=idx)
    X = pd.DataFrame(index=idx)
    train_idx = idx[: 24 * 30]
    m1 = TransformerModel(epochs=2).fit(X.loc[train_idx], price.loc[train_idx], price_history=price.loc[train_idx])
    target = X.loc[idx[24 * 30: 24 * 31]]
    p1 = m1.predict(target, price_history=price)
    tampered = price.copy()
    tampered.loc[target.index[0] - pd.Timedelta(hours=24):] += 500  # everything from t-24h on lies beyond the t-25h cutoff
    assert np.allclose(m1.predict(target.iloc[:1], price_history=tampered), p1[:1])
    m2 = TransformerModel(epochs=2).fit(X.loc[train_idx], price.loc[train_idx], price_history=price.loc[train_idx])
    assert np.allclose(m2.predict(target, price_history=price), p1, atol=1e-5)  # same seed, same weights


# --- GCN -------------------------------------------------------------------------------------------------

def test_adjacency_is_symmetric_with_the_declared_borders():
    a = zone_gnn.adjacency()
    assert np.array_equal(a, a.T) and a.sum() == 2 * len(zone_gnn.EDGES) and np.trace(a) == 0
    assert a[0].sum() == len(zone_gnn.ZONES) - 1  # DE-LU borders every other zone here


def test_normalised_adjacency_of_a_three_node_path():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    n = zone_gnn.normalised_adjacency(a)
    deg = np.array([2.0, 3.0, 2.0])  # degrees of A + I
    assert n[0, 1] == pytest.approx(1 / np.sqrt(deg[0] * deg[1])) and n[0, 0] == pytest.approx(1 / deg[0])
    assert n[0, 2] == 0 and np.allclose(n, n.T)


def test_gcn_layer_with_identity_adjacency_is_a_per_node_linear_map():
    layer = zone_gnn.GCNLayer(5, 3)
    x = torch.randn(4, 7, 5)
    assert torch.allclose(layer(x, torch.eye(7)), layer.lin(x), atol=1e-6)


def test_message_passing_moves_information_between_nodes_only_with_edges():
    x = torch.randn(1, 7, 48)
    shifted = x.clone()
    shifted[0, 3] += 5.0  # change the NL node only
    with_edges = zone_gnn.build_net("gcn").eval()
    without = zone_gnn.build_net("gcn_no_edges").eval()
    assert not torch.allclose(with_edges(x), with_edges(shifted))
    assert torch.allclose(without(x), without(shifted))  # the DE-LU readout cannot see NL
    local = zone_gnn.build_net("local_mlp").eval()
    assert torch.allclose(local(x), local(shifted))


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        zone_gnn.build_net("transformer")


def _zone_frame(days=40, seed=0):
    idx = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 70 + 25 * np.sin(2 * np.pi * (idx.hour - 7) / 24) + 8 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 7))
    return pd.DataFrame({z: base + rng.normal(0, 3, len(idx)) + 2 * i for i, z in enumerate(zone_gnn.ZONES)}, index=idx)


def test_day_tensors_use_previous_day_and_previous_week_and_skip_incomplete_days():
    h = _zone_frame(20)
    days, x, y = zone_gnn.day_tensors(h)
    assert len(days) == 13 and x.shape == (13, 7, 48) and y.shape == (13, 24)  # first 7 days have no D-7
    d = days[0]
    prev = h[h.index.date == (pd.Timestamp(d) - pd.Timedelta(days=1)).date()]
    assert np.allclose(x[0, 0, :24], prev["DE-LU"].to_numpy())
    h2 = h.drop(h.index[24 * 10 + 5])  # one missing hour makes that day unusable as a target or a lag
    assert len(zone_gnn.day_tensors(h2)[0]) < 13


def test_fit_predict_learns_a_simple_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 7, 48))
    y = x[:, 0, :24] * 2 + 1  # target is a function of the DE-LU node's previous-day prices
    pred = zone_gnn.fit_predict("local_mlp", x[:250], y[:250], x[250:], epochs=300)
    assert np.corrcoef(pred.ravel(), y[250:].ravel())[0, 1] > 0.9


def test_zone_walk_forward_does_not_use_data_after_each_cutoff():
    h = _zone_frame(160)
    start, end = h.index[24 * 130], h.index[-1]
    base = zone_gnn.walk_forward_zone(h, "flat_mlp", start, end, seed=0, epochs=20)
    tampered = h.copy()
    tampered.loc[start + pd.Timedelta(days=7):] += 300  # change everything after the first window
    after = zone_gnn.walk_forward_zone(tampered, "flat_mlp", start, end, seed=0, epochs=20)
    first = base.index < start + pd.Timedelta(days=7)
    assert first.any() and np.allclose(base[first].to_numpy(), after[after.index < start + pd.Timedelta(days=7)].to_numpy())
