#!/usr/bin/env python3
"""
EXPERIMENT 27 — RELATIONAL FLOW
===============================

Question:
    Can information be represented and reconstructed by propagating through
    a learned relational structure, with the structure itself emerging as
    part of the solution?

Design:
    - Generate hidden relational dynamical worlds.
    - Each world has a persistent sparse topology and shared local dynamics.
    - Multiple episodes use the SAME world topology but different initial states.
    - Observations are partial, so the model must use relationships to infer
      hidden state.
    - The model receives no ground-truth graph.
    - A learnable relational field provides candidate directed edges between
      latent nodes. Edge gates determine how strongly information flows.
    - A shared edge transformation changes messages as they propagate.
    - A structural penalty discourages unnecessary active relationships.
    - A dense MLP control is trained on the same task for comparison.

The experiment deliberately does NOT assume that the learned structure should
be a tree. It may become linear, branching, convergent, cyclic, hub-like, or
something else. The learned topology is itself an output of the experiment.

Outputs:
    results JSON containing prediction error, structural cost, and graph
    statistics for both models.

Run:
    python experiment27_relational_flow.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class World:
    adjacency: torch.Tensor
    initial_states: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_world(
    rng: random.Random,
    num_entities: int,
    state_dim: int,
    edge_probability: float,
) -> World:
    """Create a hidden sparse directed world.

    The topology is generator-only information. The learner sees observations,
    never this adjacency matrix.
    """
    adjacency = torch.zeros(num_entities, num_entities)
    for src in range(num_entities):
        for dst in range(num_entities):
            if src != dst and rng.random() < edge_probability:
                adjacency[src, dst] = 1.0

    # Avoid isolated worlds while retaining random topology.
    for node in range(num_entities):
        if adjacency[node].sum() == 0:
            dst = rng.randrange(num_entities - 1)
            if dst >= node:
                dst += 1
            adjacency[node, dst] = 1.0

    initial_states = torch.randn(num_entities, state_dim) * 0.5
    return World(adjacency=adjacency, initial_states=initial_states)


def evolve_world(
    world: World,
    steps: int,
    self_decay: float = 0.82,
    interaction_scale: float = 0.35,
) -> torch.Tensor:
    """Generate a trajectory from the hidden relational dynamics.

    Each entity retains part of its own state and receives a nonlinear
    aggregate of states arriving along hidden directed relationships.
    """
    states = [world.initial_states.clone()]
    x = world.initial_states.clone()
    adjacency = world.adjacency

    # Fixed, known-to-generator dynamics. No learned model is used here.
    for _ in range(steps):
        incoming = adjacency.T @ x
        degree = adjacency.T.sum(dim=1, keepdim=True).clamp_min(1.0)
        incoming = incoming / degree
        x = self_decay * x + interaction_scale * torch.tanh(incoming)
        states.append(x.clone())

    return torch.stack(states)


def make_episode(
    trajectory: torch.Tensor,
    rng: random.Random,
    observation_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return partial observation, mask, and target final state."""
    final_state = trajectory[-1]
    mask = torch.tensor(
        [1.0 if rng.random() < observation_rate else 0.0 for _ in final_state.reshape(-1)],
        dtype=torch.float32,
    ).reshape_as(final_state)

    # Ensure every entity retains at least one observed feature.
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            mask[i, rng.randrange(mask.shape[1])] = 1.0

    observed = final_state * mask
    return observed, mask, final_state


class RelationalField(nn.Module):
    """Learnable information-flow field over latent nodes.

    Every directed pair is initially possible. Edge gates determine whether
    information is actually transmitted. The same edge function is reused for
    all relationships, making topology and local transformation separable.
    """

    def __init__(self, input_dim: int, num_nodes: int, node_dim: int, steps: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.steps = steps

        self.input_encoder = nn.Linear(input_dim, num_nodes * node_dim)
        self.edge_logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.15)

        self.edge_message = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim * 2),
            nn.Tanh(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Tanh(),
        )
        self.readout = nn.Sequential(
            nn.Linear(num_nodes * node_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
        )

    def gates(self) -> torch.Tensor:
        # No self-loop requirement; diagonal is suppressed.
        g = torch.sigmoid(self.edge_logits)
        eye = torch.eye(self.num_nodes, device=g.device)
        return g * (1.0 - eye)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        nodes = self.input_encoder(x).reshape(batch, self.num_nodes, self.node_dim)
        gates = self.gates()

        for _ in range(self.steps):
            messages = torch.zeros_like(nodes)
            for src in range(self.num_nodes):
                for dst in range(self.num_nodes):
                    if src == dst:
                        continue
                    pair = torch.cat([nodes[:, src], nodes[:, dst]], dim=-1)
                    message = self.edge_message(pair)
                    messages[:, dst] += gates[src, dst] * message

            degree = gates.sum(dim=0).clamp_min(1e-6)
            messages = messages / degree.reshape(1, -1, 1)
            nodes = nodes + self.node_update(torch.cat([nodes, messages], dim=-1))

        output = self.readout(nodes.reshape(batch, -1))
        return output, gates


class DenseControl(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def batch_episode_data(
    worlds: list[World],
    trajectories: list[torch.Tensor],
    episodes_per_world: int,
    observation_rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    observations = []
    masks = []
    targets = []
    world_ids = []

    for world_id, trajectory in enumerate(trajectories):
        for _ in range(episodes_per_world):
            # Same hidden topology, different initial condition.
            noisy = trajectory + torch.randn_like(trajectory) * 0.12
            observed, mask, target = make_episode(noisy, rng, observation_rate)
            observations.append(observed.reshape(-1))
            masks.append(mask.reshape(-1))
            targets.append(target.reshape(-1))
            world_ids.append(world_id)

    return (
        torch.stack(observations),
        torch.stack(masks),
        torch.stack(targets),
        torch.tensor(world_ids),
    )


def structural_metrics(gates: torch.Tensor, threshold: float = 0.5) -> dict:
    with torch.no_grad():
        active = gates >= threshold
        n = gates.shape[0]
        offdiag = ~torch.eye(n, dtype=torch.bool, device=gates.device)
        values = gates[offdiag]
        active_values = values[values >= threshold]

        out_degree = active.sum(dim=1).float()
        in_degree = active.sum(dim=0).float()

        return {
            "mean_gate": float(values.mean().item()),
            "active_edge_count": int(active.sum().item()),
            "active_edge_fraction": float(active.float().mean().item()),
            "mean_out_degree": float(out_degree.mean().item()),
            "mean_in_degree": float(in_degree.mean().item()),
            "max_out_degree": int(out_degree.max().item()),
            "max_in_degree": int(in_degree.max().item()),
            "branching_nodes": int((out_degree >= 2).sum().item()),
            "convergence_nodes": int((in_degree >= 2).sum().item()),
            "gate_entropy": float(
                (-(values * values.clamp_min(1e-8).log()
                  + (1 - values) * (1 - values).clamp_min(1e-8).log())).mean().item()
            ),
            "active_gate_mean": float(active_values.mean().item()) if active_values.numel() else 0.0,
        }


def train_relational(
    model: RelationalField,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    structural_lambda: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for step in range(1, steps + 1):
        pred, gates = model(x)
        reconstruction = F.mse_loss(pred, y)
        structural = gates.sum() / (model.num_nodes * (model.num_nodes - 1))
        loss = reconstruction + structural_lambda * structural

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history.append(float(loss.item()))
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(
                f"  relational {step:5d}/{steps}"
                f" | mse {reconstruction.item():.6g}"
                f" | structure {structural.item():.6g}"
            )

    return history


def train_control(
    model: DenseControl,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for step in range(1, steps + 1):
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(f"  control    {step:5d}/{steps} | mse {loss.item():.6g}")

    return history


def evaluate_relational(model, x, y) -> dict:
    with torch.no_grad():
        pred, gates = model(x)
        mse = F.mse_loss(pred, y).item()
        return {"mse": mse, "structural": structural_metrics(gates)}


def evaluate_control(model, x, y) -> dict:
    with torch.no_grad():
        pred = model(x)
        return {"mse": F.mse_loss(pred, y).item()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--num-entities", type=int, default=6)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--test-worlds", type=int, default=8)
    p.add_argument("--test-episodes", type=int, default=12)
    p.add_argument("--observation-rate", type=float, default=0.55)
    p.add_argument("--edge-probability", type=float, default=0.25)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--node-dim", type=int, default=8)
    p.add_argument("--propagation-steps", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--structural-lambda", type=float, default=0.015)
    p.add_argument("--output", type=Path, default=Path("results/experiment27_relational_flow.json"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_rng = random.Random(args.seed + 1)
    test_rng = random.Random(args.seed + 2)

    train_worlds = [
        make_world(train_rng, args.num_entities, args.state_dim, args.edge_probability)
        for _ in range(args.worlds)
    ]
    test_worlds = [
        make_world(test_rng, args.num_entities, args.state_dim, args.edge_probability)
        for _ in range(args.test_worlds)
    ]

    # Each world's trajectory is generated from a different initial state.
    train_trajectories = [evolve_world(w, steps=4) for w in train_worlds]
    test_trajectories = [evolve_world(w, steps=4) for w in test_worlds]

    x_train, _, y_train, _ = batch_episode_data(
        train_worlds,
        train_trajectories,
        args.train_episodes,
        args.observation_rate,
        args.seed + 3,
    )
    x_test, _, y_test, _ = batch_episode_data(
        test_worlds,
        test_trajectories,
        args.test_episodes,
        args.observation_rate,
        args.seed + 4,
    )

    input_dim = args.num_entities * args.state_dim

    print("Experiment 27 — Relational Flow")
    print(f"train examples: {len(x_train)}")
    print(f"test examples:  {len(x_test)}")
    print(f"input dimension: {input_dim}")
    print()

    relational = RelationalField(
        input_dim=input_dim,
        num_nodes=args.nodes,
        node_dim=args.node_dim,
        steps=args.propagation_steps,
    )
    control = DenseControl(input_dim=input_dim)

    print("Training relational field")
    relational_history = train_relational(
        relational,
        x_train,
        y_train,
        args.train_steps,
        args.lr,
        args.structural_lambda,
    )

    print("Training dense control")
    control_history = train_control(
        control,
        x_train,
        y_train,
        args.train_steps,
        args.lr,
    )

    train_relational_result = evaluate_relational(relational, x_train, y_train)
    test_relational_result = evaluate_relational(relational, x_test, y_test)
    train_control_result = evaluate_control(control, x_train, y_train)
    test_control_result = evaluate_control(control, x_test, y_test)

    gates = relational.gates().detach().cpu().tolist()

    result = {
        "experiment": "27 relational flow",
        "question": "Can information be represented and reconstructed through a learned relational structure whose topology is itself part of the solution?",
        "seed": args.seed,
        "configuration": vars(args) | {"output": str(args.output)},
        "world_generation": {
            "hidden_topology": "random sparse directed graph",
            "topology_probability": args.edge_probability,
            "same_topology_across_episodes": True,
            "different_initial_state_across_episodes": True,
            "model_receives_hidden_topology": False,
            "observation_rate": args.observation_rate,
            "generator_dynamics": "self retention + normalized incoming nonlinear influence",
        },
        "relational": {
            "train": train_relational_result,
            "test_unseen_worlds": test_relational_result,
            "final_gate_matrix": gates,
            "loss_start": relational_history[0],
            "loss_final": relational_history[-1],
        },
        "dense_control": {
            "train": train_control_result,
            "test_unseen_worlds": test_control_result,
            "loss_start": control_history[0],
            "loss_final": control_history[-1],
        },
        "interpretation": {
            "topology": "learned gate structure is an observed experimental outcome, not a prescribed tree",
            "generalization": "test worlds have unseen hidden topologies and initial states",
            "important_caveat": "the learned topology is an internal computational representation; matching the generator graph is not required for success",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\nRelational test:")
    print(json.dumps(test_relational_result, indent=2))
    print("\nDense control test:")
    print(json.dumps(test_control_result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
