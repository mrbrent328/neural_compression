"""Structural validation for E41.  These tests validate the information boundary
and experiment construction; they do not predeclare the research result.
"""
import torch

from E41_relational_concept_memory import Config, make_world, split_relations


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


def test_world_contains_related_distinct_tokens():
    cfg = Config()
    world = make_world(cfg)
    found = False
    for f in range(cfg.families):
        ids = torch.where(world.family == f)[0].tolist()
        if len({int(world.tokens[i]) for i in ids}) > 1:
            found = True
            break
    assert found, "world must contain related concepts with distinct surface tokens"


def test_relation_split_is_disjoint_and_complete():
    cfg = Config()
    world = make_world(cfg)
    train, test = split_relations(world, cfg.seed)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(world.relations)


def main():
    tests = [
        test_world_contains_surface_ambiguity,
        test_world_contains_related_distinct_tokens,
        test_relation_split_is_disjoint_and_complete,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
