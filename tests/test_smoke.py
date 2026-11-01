"""Fast smoke tests on a tiny random graph (no dataset download).

    pytest -q tests/
"""
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masd import MODELS, POLICIES, build_model, train_masd, train_supervised  # noqa: E402
from masd.losses import masd_loss  # noqa: E402
from masd.metrics import CKA  # noqa: E402
from masd.uncertainty import node_uncertainty  # noqa: E402


def tiny_graph(n=60, f=16, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=g)
    edge_index = torch.randint(0, n, (2, 4 * n), generator=g)
    y = torch.randint(0, c, (n,), generator=g)
    idx = torch.randperm(n, generator=g)
    masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    masks[0][idx[:20]] = True
    masks[1][idx[20:35]] = True
    masks[2][idx[35:]] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=masks[0], val_mask=masks[1], test_mask=masks[2]), f, c


@pytest.mark.parametrize("name", sorted(MODELS))
def test_models_output_log_probs(name):
    data, f, c = tiny_graph()
    m = build_model(name, f, c, hidden=8, heads=2)
    out = m(data.x, data.edge_index)
    assert out.shape == (data.num_nodes, c)
    assert torch.allclose(out.exp().sum(-1), torch.ones(data.num_nodes), atol=1e-5)
    assert m.get_embeddings(data.x, data.edge_index).size(0) == data.num_nodes


@pytest.mark.parametrize("kind", ["entropy", "mc_dropout", "margin", "least_confidence"])
def test_uncertainty_in_unit_interval(kind):
    data, f, c = tiny_graph()
    m = build_model("gcn", f, c, hidden=8)
    u = node_uncertainty(kind, m, data.x, data.edge_index, m(data.x, data.edge_index).detach())
    assert u.shape == (data.num_nodes,) and (u >= 0).all() and (u <= 1 + 1e-6).all()


@pytest.mark.parametrize("impl", ["paper", "notebook"])
def test_loss_is_finite_and_differentiable(impl):
    data, f, c = tiny_graph()
    s, t = build_model("gcn", f, c, hidden=8), build_model("gcn", f, c, hidden=8)
    s_lp = s(data.x, data.edge_index)
    with torch.no_grad():
        t_lp = t(data.x, data.edge_index)
    u = torch.rand(data.num_nodes)
    loss, _ = masd_loss(s_lp, t_lp, None, data.y, data.train_mask, u, 1.0, 0.1, 0.5, 10.0, impl=impl)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("policy", POLICIES)
def test_masd_runs_every_policy(policy):
    data, f, c = tiny_graph()
    teacher, _ = train_supervised(build_model("gcn", f, c, hidden=8), data, epochs=5, verbose=False)
    student, memory, hist = train_masd(build_model("gcn", f, c, hidden=8), teacher, data,
                                       memory_policy=policy, epochs=6, patience=None, verbose=False)
    assert len(hist) == 6 and memory.ready


def test_cka_identity_is_one():
    x = torch.randn(30, 5)
    assert abs(CKA.linear(x, x) - 1.0) < 1e-6
