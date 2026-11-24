"""Small deterministic integrity checks for E42."""
import torch

from E42_scaled_modular_concept_network import (
    DenseBaseline, ModularConceptNetwork, build_context, make_graph
)


def test_graph_split_and_context():
    train, test = make_graph(256, 8, 4, 7)
    assert train.shape[1] == 3 and test.shape[1] == 3
    ctx = build_context(256, train, 6, 8)
    assert ctx[0].shape == (256, 6)
    assert ctx[1].shape == (256, 6)
    assert ctx[2].shape == (256, 6)
    assert len(set(map(tuple, train.tolist())) & set(map(tuple, test.tolist()))) == 0


def test_models_return_finite_scores():
    n, rels, dim, batch, k = 64, 8, 16, 5, 4
    src = torch.arange(batch)
    rel = torch.zeros(batch, dtype=torch.long)
    ctx_ids = torch.randint(0, n, (batch, k))
    ctx_rels = torch.randint(0, rels, (batch, k))
    mask = torch.ones(batch, k, dtype=torch.bool)
    candidates = torch.randint(0, n, (batch, 9))
    for model in (DenseBaseline(n, rels, dim),
                  ModularConceptNetwork(n, rels, dim, True),
                  ModularConceptNetwork(n, rels, dim, False)):
        q = model.encode(src, rel, ctx_ids, ctx_rels, mask)
        scores = model.score(q, candidates)
        assert scores.shape == (batch, 9)
        assert torch.isfinite(scores).all()
