"""Graph neural network over interconnected bidding zones, written from scratch (no torch_geometric).

Question: does the interconnection graph help forecast DE-LU day-ahead prices beyond simply knowing the neighbours'
prices? Nodes are bidding zones, edges are physical borders (plus a few neighbour-neighbour borders). Each node's
features are its own prices of the previous day and of the same day a week earlier (48 numbers, all known at the
day-ahead cutoff); the target is the 24 hourly DE-LU prices of the day. Three architectures see the same information:

    local_mlp    DE-LU features only (no neighbour information)
    flat_mlp     every zone's features concatenated (neighbour information, no graph structure)
    gcn          two graph-convolution layers over the zone graph, read out at the DE-LU node
    gcn_no_edges the same GCN with the adjacency replaced by the identity (a control: same layers, no message passing)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ZONES = ["DE-LU", "AT", "FR", "NL", "CH", "PL", "CZ"]
EDGES = [("DE-LU", z) for z in ZONES[1:]] + [("AT", "CH"), ("AT", "CZ"), ("CZ", "PL"), ("FR", "CH")]
LAGS_DAYS = (1, 7)  # features: the zone's prices D-1 and D-7 (24 hourly values each)


def adjacency(edges=EDGES, zones=ZONES) -> np.ndarray:
    idx = {z: i for i, z in enumerate(zones)}
    a = np.zeros((len(zones), len(zones)))
    for u, v in edges:
        a[idx[u], idx[v]] = a[idx[v], idx[u]] = 1.0
    return a


def normalised_adjacency(a: np.ndarray) -> np.ndarray:
    """Kipf-Welling propagation matrix D^-1/2 (A + I) D^-1/2."""
    a_hat = a + np.eye(len(a))
    d = a_hat.sum(axis=1) ** -0.5
    return a_hat * d[:, None] * d[None, :]


class GCNLayer(nn.Module):
    """H' = A_hat H W + b, with H shaped (batch, nodes, features)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return a_hat @ self.lin(h)  # (nodes, nodes) @ (batch, nodes, out) broadcasts over the batch


class ZoneGCN(nn.Module):
    def __init__(self, a_hat: np.ndarray, n_feat: int = 48, hidden: int = 32, target_node: int = 0, n_out: int = 24):
        super().__init__()
        self.register_buffer("a_hat", torch.tensor(a_hat, dtype=torch.float32))
        self.l1, self.l2 = GCNLayer(n_feat, hidden), GCNLayer(hidden, hidden)
        self.readout = nn.Linear(hidden, n_out)
        self.target = target_node

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, nodes, features)
        h = torch.relu(self.l1(x, self.a_hat))
        h = torch.relu(self.l2(h, self.a_hat))
        return self.readout(h[:, self.target, :])


class FlatMLP(nn.Module):
    def __init__(self, n_nodes: int, n_feat: int = 48, hidden: int = 64, n_out: int = 24, use_nodes: int | None = None):
        super().__init__()
        self.use = use_nodes
        n_in = (use_nodes or n_nodes) * n_feat
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.use, :] if self.use else x
        return self.net(x.flatten(1))


def build_net(arch: str, n_nodes: int = len(ZONES)) -> nn.Module:
    if arch == "local_mlp":
        return FlatMLP(n_nodes, use_nodes=1)  # node 0 is DE-LU
    if arch == "flat_mlp":
        return FlatMLP(n_nodes)
    if arch == "gcn":
        return ZoneGCN(normalised_adjacency(adjacency()))
    if arch == "gcn_no_edges":
        return ZoneGCN(np.eye(n_nodes))
    raise ValueError(f"unknown architecture {arch!r}")


ARCHS = ["local_mlp", "flat_mlp", "gcn_no_edges", "gcn"]


def day_tensors(hourly: pd.DataFrame, zones=ZONES) -> tuple[list, np.ndarray, np.ndarray]:
    """(days, X, Y): X[i] is (zones, 48) features for day days[i], Y[i] the 24 DE-LU prices. Only complete UTC days
    whose D-1 and D-7 are also complete are kept."""
    h = hourly[list(zones)]
    by_day = {d: g for d, g in h.groupby(h.index.date) if len(g) == 24}
    days, xs, ys = [], [], []
    for d in sorted(by_day):
        prev = [d - pd.Timedelta(days=k) for k in LAGS_DAYS]
        prev = [p.date() if hasattr(p, "date") else p for p in prev]
        if not all(p in by_day for p in prev):
            continue
        feats = np.concatenate([by_day[p].to_numpy().T for p in prev], axis=1)  # (zones, 48)
        days.append(d)
        xs.append(feats)
        ys.append(by_day[d][zones[0]].to_numpy())
    return days, np.stack(xs), np.stack(ys)


def fit_predict(arch: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int = 0,
                epochs: int = 150, lr: float = 3e-3, weight_decay: float = 1e-4) -> np.ndarray:
    """Full-batch Adam on standardised data; returns predictions for x_test in EUR/MWh."""
    torch.manual_seed(seed)
    mu, sd = x_train.mean(), x_train.std() + 1e-8
    ymu, ysd = y_train.mean(), y_train.std() + 1e-8
    xt = torch.tensor((x_train - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((y_train - ymu) / ysd, dtype=torch.float32)
    net = build_net(arch, x_train.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor((x_test - mu) / sd, dtype=torch.float32)).numpy()
    return out * ysd + ymu


def walk_forward_zone(hourly: pd.DataFrame, arch: str, test_start: pd.Timestamp, end: pd.Timestamp,
                      retrain_every_days: int = 7, seed: int = 0, **fit_kw) -> pd.Series:
    """Same protocol as src.walk_forward: every `retrain_every_days`, train on complete days that end before the
    cutoff, predict the days of the next window. Returns hourly DE-LU forecasts indexed like the target hours."""
    days, x, y = day_tensors(hourly)
    day_ts = pd.to_datetime(pd.Series(days)).dt.tz_localize("UTC").to_numpy()
    out: list[pd.Series] = []
    cutoff = test_start
    while cutoff < end:
        window_end = min(cutoff + pd.Timedelta(days=retrain_every_days), end)
        train = np.array([pd.Timestamp(t) + pd.Timedelta(days=1) <= cutoff for t in day_ts])
        test = np.array([(pd.Timestamp(t) + pd.Timedelta(days=1) > cutoff) and (pd.Timestamp(t) < window_end) for t in day_ts])
        if train.sum() >= 120 and test.any():
            pred = fit_predict(arch, x[train], y[train], x[test], seed=seed, **fit_kw)
            for t, row in zip(day_ts[test], pred):
                idx = pd.date_range(pd.Timestamp(t), periods=24, freq="h")
                idx = idx[(idx >= cutoff) & (idx < window_end)]
                out.append(pd.Series(row[[i.hour for i in idx]], index=idx))
        cutoff = window_end
    return pd.concat(out).sort_index()
