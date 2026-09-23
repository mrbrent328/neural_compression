"""Structural validation for E41.

These tests validate the information boundary and experimental controls. They
do not predeclare the scientific result.
"""
import torch

from E41_relational_concept_memory import (
    BasePredictor,
    Config,
    MemoryAidedPredictor,
    ambiguous_primary_tokens,
    make_world,
    query_feature,
    split_relations,
    one,
)


def test_world_contains_surface_ambiguity():
    cfg = Config()
    world = make_world(cfg)
    groups = {}
    for i, token in enumerate(world.tokens.tolist()):
        groups.setdefault(token, []).append(i)
    assert any(
        len(ids) > 1 and len({int(world.family[i]) for i in ids}) > 1
        for ids in groups.values()
    ), "world must contain same-token/different-family homonyms"


def test_world_contains_true_aliases():
    cfg = Config()
    world = make_world(cfg)
    aliased = torch.where(world.aliases >= 0)[0].tolist()
    assert aliased, "world must contain concepts with alternate surface labels"
    assert all(
        int(world.aliases[i]) != int(world.tokens[i]) for i in aliased
    ), "an alias must differ from the primary token for the same concept"


def test_relation_split_is_disjoint_and_complete():
    cfg = Config()
    world = make_world(cfg)
    train, test = split_relations(world, cfg.seed)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(world.relations)


def test_masked_homonyms_have_identical_local_evidence():
    cfg = Config()
    world = make_world(cfg)
    ambiguous = ambiguous_primary_tokens(world)

    pair = None
    for token in ambiguous:
        ids = [
            i for i, t in enumerate(world.tokens.tolist()) if int(t) == token
        ]
        for a in ids:
            for b in ids:
                if a < b and int(world.family[a]) != int(world.family[b]):
                    pair = (a, b)
                    break
            if pair:
                break
        if pair:
            break

    assert pair is not None
    a, b = pair
    assert int(world.tokens[a]) == int(world.tokens[b])
    assert torch.equal(query_feature(world, a, True), query_feature(world, b, True))
    assert torch.count_nonzero(query_feature(world, a, True)) == 0


def test_empty_memory_is_exact_base_control():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    world = make_world(cfg)
    base = BasePredictor(cfg)
    aided = MemoryAidedPredictor(cfg, base)
    state = aided.memory.empty()

    src, relation_type, _ = world.relations[0]
    token = one(int(world.tokens[src]))
    feature = world.features[src]
    relation = one(relation_type)

    with torch.no_grad():
        base_pred, _ = base(token, feature, relation)
        aided_pred, frozen_base_pred, _, recalled, _ = aided.predict(
            state, token, feature, relation
        )

    assert torch.equal(recalled, torch.zeros_like(recalled))
    assert torch.allclose(base_pred, frozen_base_pred, atol=0.0, rtol=0.0)
    assert torch.allclose(base_pred, aided_pred, atol=1e-7, rtol=1e-7)


def test_base_is_frozen_inside_memory_model():
    cfg = Config()
    base = BasePredictor(cfg)
    aided = MemoryAidedPredictor(cfg, base)
    assert all(not p.requires_grad for p in aided.base.parameters())
    assert any(p.requires_grad for p in aided.memory.parameters())
    assert any(p.requires_grad for p in aided.aid.parameters())


def main():
    tests = [
        test_world_contains_surface_ambiguity,
        test_world_contains_true_aliases,
        test_relation_split_is_disjoint_and_complete,
        test_masked_homonyms_have_identical_local_evidence,
        test_empty_memory_is_exact_base_control,
        test_base_is_frozen_inside_memory_model,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
