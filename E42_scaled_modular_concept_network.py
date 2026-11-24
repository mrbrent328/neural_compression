#!/usr/bin/env python3
"""E42: scaled modular concept network.

Compares a shared embedding baseline, a modular concept model with shared
relation-aware message processing, and the same modular model with messages
disabled. Each concept has a persistent learned state; shared machinery
processes neighbor states and a query-conditioned router aggregates context.

Synthetic family IDs generate graph regularities but are never model inputs.
This is an engineering test of modular relational learning, not a language
model or consciousness experiment.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_graph(n, relation_types, edges_per_node, seed):
    rng = np.random.default_rng(seed)
    family_count = max(32, n // 128)
    family = rng.integers(0, family_count, size=n, dtype=np.int64)
    sources = np.repeat(np.arange(n, dtype=np.int64), edges_per_node)
    rels = rng.integers(0, relation_types, size=len(sources), dtype=np.int64)
    dests = np.empty(len(sources), dtype=np.int64)
    members = [np.flatnonzero(family == f) for f in range(family_count)]
    for i, (src, rel) in enumerate(zip(sources, rels)):
        if rel % 2 == 0:
            pool = members[family[src]]
        else:
            target_family = (family[src] + 1 + int(rng.integers(0, family_count - 1))) % family_count
            pool = members[target_family]
        if len(pool) <= 1:
            dst = int(rng.integers(n))
        else:
            dst = int(pool[int(rng.integers(len(pool)))])
            while dst == src:
                dst = int(pool[int(rng.integers(len(pool)))])
        dests[i] = dst
    order = rng.permutation(len(sources))
    cut = int(0.9 * len(order))
    train = np.stack([sources[order[:cut]], rels[order[:cut]], dests[order[:cut]]], axis=1)
    test = np.stack([sources[order[cut:]], rels[order[cut:]], dests[order[cut:]]], axis=1)
    return train, test


def build_context(n, train_edges, max_neighbors, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n)]
    for src, rel, dst in train_edges:
        buckets[int(src)].append((int(dst), int(rel)))
    ids = np.zeros((n, max_neighbors), dtype=np.int64)
    rels = np.zeros((n, max_neighbors), dtype=np.int64)
    mask = np.zeros((n, max_neighbors), dtype=np.bool_)
    for src, items in enumerate(buckets):
        if len(items) > max_neighbors:
            chosen = rng.choice(len(items), size=max_neighbors, replace=False)
            items = [items[int(i)] for i in chosen]
        for j, (dst, rel) in enumerate(items):
            ids[src, j], rels[src, j], mask[src, j] = dst, rel, True
    return ids, rels, mask


class DenseBaseline(nn.Module):
    def __init__(self, n, relation_types, dim):
        super().__init__()
        self.nodes = nn.Embedding(n, dim)
        self.relations = nn.Embedding(relation_types, dim)
        self.query = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        nn.init.normal_(self.nodes.weight, std=0.08)
        nn.init.normal_(self.relations.weight, std=0.08)

    def encode(self, src, rel, ctx_ids=None, ctx_rels=None, ctx_mask=None):
        return self.query(torch.cat([self.nodes(src), self.relations(rel)], dim=-1))

    def score(self, query, candidates):
        return (query[:, None, :] * self.nodes(candidates)).sum(-1) / math.sqrt(query.shape[-1])


class ModularConceptNetwork(DenseBaseline):
    def __init__(self, n, relation_types, dim, use_messages=True):
        super().__init__(n, relation_types, dim)
        self.use_messages = use_messages
        self.local_processor = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.router_query = nn.Linear(dim * 2, dim, bias=False)
        self.message_key = nn.Linear(dim, dim, bias=False)
        self.message_value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))

    def encode(self, src, rel, ctx_ids, ctx_rels, ctx_mask):
        own = super().encode(src, rel)
        if not self.use_messages or ctx_ids is None:
            return own
        neighbor = self.nodes(ctx_ids)
        neighbor_rel = self.relations(ctx_rels)
        messages = self.local_processor(torch.cat([neighbor, neighbor_rel], dim=-1))
        q = self.router_query(torch.cat([self.nodes(src), self.relations(rel)], dim=-1))
        logits = (self.message_key(messages) * q[:, None, :]).sum(-1) / math.sqrt(q.shape[-1])
        logits = logits.masked_fill(~ctx_mask, -1e9)
        weights = torch.softmax(logits, dim=-1) * ctx_mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        context = (weights[:, :, None] * self.message_value(messages)).sum(1)
        gate = self.gate(torch.cat([own, context], dim=-1))
        return self.fusion(torch.cat([own, gate * context], dim=-1))


def batch_tensors(batch, ctx, device):
    src = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
    rel = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
    dst = torch.as_tensor(batch[:, 2], dtype=torch.long, device=device)
    ci, cr, cm = ctx
    return (src, rel, dst,
            torch.as_tensor(ci[batch[:, 0]], dtype=torch.long, device=device),
            torch.as_tensor(cr[batch[:, 0]], dtype=torch.long, device=device),
            torch.as_tensor(cm[batch[:, 0]], dtype=torch.bool, device=device))


def train_model(model, train_edges, ctx, n, device, args, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.train()
    losses = []
    started = time.perf_counter()
    for step in range(args.steps):
        batch = train_edges[rng.integers(0, len(train_edges), size=args.batch_size)]
        src, rel, dst, ci, cr, cm = batch_tensors(batch, ctx, device)
        neg_np = rng.integers(0, n, size=(len(batch), args.negatives))
        collisions = neg_np == batch[:, 2:3]
        neg_np[collisions] = (neg_np[collisions] + 1) % n
        neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
        candidates = torch.cat([dst[:, None], neg], dim=1)
        scores = model.score(model.encode(src, rel, ci, cr, cm), candidates)
        labels = torch.zeros_like(scores)
        labels[:, 0] = 1.0
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 20) == 0:
            losses.append(float(loss.detach().cpu()))
    return {"seconds": time.perf_counter() - started, "sampled_loss": losses,
            "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def evaluate(model, test_edges, ctx, n, device, args, seed):
    rng = np.random.default_rng(seed)
    if len(test_edges) > args.max_eval_queries:
        test_edges = test_edges[rng.choice(len(test_edges), args.max_eval_queries, replace=False)]
    model.eval()
    ranks = []
    for start in range(0, len(test_edges), 512):
        batch = test_edges[start:start + 512]
        src, rel, dst, ci, cr, cm = batch_tensors(batch, ctx, device)
        neg = rng.integers(0, n, size=(len(batch), args.eval_negatives))
        collision = neg == batch[:, 2:3]
        neg[collision] = (neg[collision] + 1) % n
        candidates = torch.as_tensor(np.concatenate([batch[:, 2:3], neg], axis=1),
                                     dtype=torch.long, device=device)
        scores = model.score(model.encode(src, rel, ci, cr, cm), candidates)
        rank = 1 + (scores[:, 1:] >= scores[:, :1]).sum(dim=1)
        ranks.extend(rank.cpu().tolist())
    r = np.asarray(ranks, dtype=np.float64)
    return {"queries": len(r), "mrr": float(np.mean(1.0 / r)),
            "hits_at_1": float(np.mean(r <= 1)),
            "hits_at_10": float(np.mean(r <= 10)), "mean_rank": float(np.mean(r))}


def run_one(n, seed, args, device, out_dir):
    seed_all(seed)
    train, test = make_graph(n, args.relation_types, args.edges_per_node, seed)
    ctx = build_context(n, train, args.max_neighbors, seed + 10000)
    result = {"n_concepts": n, "seed": seed, "n_train_edges": len(train),
              "n_test_edges": len(test), "device": str(device), "models": {}}
    factories = {
        "dense_baseline": lambda: DenseBaseline(n, args.relation_types, args.dim),
        "modular": lambda: ModularConceptNetwork(n, args.relation_types, args.dim, True),
        "modular_no_messages": lambda: ModularConceptNetwork(n, args.relation_types, args.dim, False)}
    for name, factory in factories.items():
        model = factory().to(device)
        train_metrics = train_model(model, train, ctx, n, device, args, seed + len(name))
        test_metrics = evaluate(model, test, ctx, n, device, args, seed + 20000)
        result["models"][name] = {"train": train_metrics, "test": test_metrics}
        torch.save({"model": model.state_dict(), "n": n, "seed": seed,
                    "model_name": name, "args": vars(args)},
                   out_dir / f"e42_n{n}_seed{seed}_{name}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scales", nargs="+", type=int, default=[8192, 32768, 65536])
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    p.add_argument("--relation-types", type=int, default=16)
    p.add_argument("--edges-per-node", type=int, default=8)
    p.add_argument("--max-neighbors", type=int, default=12)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--negatives", type=int, default=16)
    p.add_argument("--eval-negatives", type=int, default=64)
    p.add_argument("--max-eval-queries", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output", default="results/e42")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm requested but torch.cuda.is_available() is false.")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "E42_scaled_modular_concept_network",
        "hypothesis": "Shared local processing plus selective relational messages improves compositional link prediction as concept count scales.",
        "limitations": [
            "Synthetic graph only; no natural language in this first experiment.",
            "Uniform negative sampling; results depend on candidate protocol.",
            "Shared processor over persistent concept states is a modular prototype, not one separately parameterized MLP per concept.",
            "Parameter count and wall-clock measurements must be interpreted alongside the matched controls."],
        "args": vars(args), "device": str(device), "runs": []}
    result_path = out_dir / "e42_results.json"
    for n in args.scales:
        for seed in args.seeds:
            print(f"\\n=== E42 concepts={n:,} seed={seed} device={device} ===", flush=True)
            result = run_one(n, seed, args, device, out_dir)
            manifest["runs"].append(result)
            result_path.write_text(json.dumps(manifest, indent=2))
            for name, metrics in result["models"].items():
                print(name, metrics["test"], flush=True)
    result_path.write_text(json.dumps(manifest, indent=2))
    print(f"\\nSaved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
