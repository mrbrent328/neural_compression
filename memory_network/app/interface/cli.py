from __future__ import annotations
import argparse
import shlex
from ..system.memory_system import NeuralMemorySystem

def main():
    parser = argparse.ArgumentParser(description="Neural Compression working relational memory")
    parser.add_argument("--db", default="data/memory.db")
    args = parser.parse_args()
    memory = NeuralMemorySystem(args.db)
    print("Neural Memory — remember A REL B | inspect A | query A [REL] | train N | quit")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"quit", "exit"}:
                break
            parts = shlex.split(line)
            try:
                if parts[0] == "remember" and len(parts) == 4:
                    obs = memory.remember(parts[1], parts[2], parts[3])
                    print(f"remembered: {parts[1]} {parts[2]} {parts[3]} (observation {obs.id})")
                elif parts[0] == "inspect" and len(parts) == 2:
                    result = memory.inspect(parts[1])
                    if result is None:
                        print("not found")
                    else:
                        print(f"{result['entity'].name} [{result['entity'].entity_type}]")
                        for direction, relation, target in result["edges"]:
                            print(f"  {direction} {relation} {target}")
                        print(f"  context_norm={sum(x*x for x in result['context']) ** 0.5:.4f}")
                elif parts[0] == "query" and len(parts) in (2, 3):
                    for score, relation, target in memory.query(parts[1], parts[2] if len(parts) == 3 else None):
                        print(f"{score:+.3f}  {relation}  {target}")
                elif parts[0] == "train":
                    steps = int(parts[1]) if len(parts) > 1 else 500
                    print(memory.train(steps))
                else:
                    print("usage: remember A REL B | inspect A | query A [REL] | train N | quit")
            except Exception as exc:
                print(f"error: {exc}")
    finally:
        memory.close()

if __name__ == "__main__":
    main()
