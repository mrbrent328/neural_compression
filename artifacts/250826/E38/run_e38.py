#!/usr/bin/env python3
"""E38 end-to-end relational-assisted local inference experiment."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from e38_corpus import build
from e38_memory import RelationalMemory
from e38_model import LlamaCppBackend, LMStudioBackend

OUT = Path(__file__).with_name("results")
OUT.mkdir(exist_ok=True)


def ingest_fact(memory: RelationalMemory, text: str) -> None:
    m = re.fullmatch(r"(\w+) owns the (.+)\.", text)
    if m:
        memory.observe_relation(m.group(1), "owns", m.group(2))
        return
    m = re.fullmatch(r"The (.+) is at the (.+)\.", text)
    if m:
        memory.observe_relation(m.group(1), "at", m.group(2))


def question_entities(text: str) -> list[str]:
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    return words[:1]


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def prompt_baseline(history: list[str], question: str) -> str:
    return "You answer questions using only the supplied history. Give the shortest correct answer.\n\nHISTORY:\n" + "\n".join(history) + f"\n\nQUESTION: {question}\nANSWER:"


def prompt_relation(context: str, question: str) -> str:
    return "You answer questions using only the supplied relational facts. Give the shortest correct answer.\n\nRELATIONAL MEMORY:\n" + context + f"\n\nQUESTION: {question}\nANSWER:"


def choose_backend(args):
    if args.backend == "llama":
        if not args.model:
            raise SystemExit("--model is required for --backend llama")
        return LlamaCppBackend(args.model, args.executable, args.context, args.max_tokens)
    return LMStudioBackend(args.url, args.model or "local-model", args.max_tokens)


def run(args):
    facts, questions = build(args.seed, args.facts, args.questions)
    memory = RelationalMemory(args.db)
    history: list[str] = []
    backend = choose_backend(args)
    results = []

    # The stream is observed first. ``facts`` is the number of observations;
    # the corpus expands each observation into two textual fact items.
    for item in facts:
        history.append(item.text)
        ingest_fact(memory, item.text)

    stats = memory.stats()
    for i, item in enumerate(questions):
        entities = question_entities(item.text)
        context = memory.serialize_context(entities, depth=2, limit=args.relation_limit)

        for condition, prompt in (
            ("raw_history", prompt_baseline(history, item.text)),
            ("relational", prompt_relation(context, item.text)),
        ):
            start = time.perf_counter()
            generation = backend.generate(prompt)
            elapsed = time.perf_counter() - start
            answer = normalize(generation.text)
            expected = normalize(item.answer or "")
            results.append({
                "question_id": i,
                "condition": condition,
                "question": item.text,
                "expected": item.answer,
                "answer": generation.text,
                "correct": expected in answer,
                "latency_s": elapsed,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "context_chars": len(prompt),
                "memory_context_chars": len(context),
            })

    payload = {
        "experiment": "E38 — Relational-Assisted Local Inference",
        "seed": args.seed,
        "fact_observations": args.facts,
        "fact_items": len(facts),
        "questions_requested": args.questions,
        "questions": len(questions),
        "memory": stats,
        "conditions": ["raw_history", "relational"],
        "results": results,
    }
    path = OUT / "e38_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    memory.close()
    print(json.dumps({"output": str(path), "memory": stats, "fact_observations": args.facts, "fact_items": len(facts), "questions": len(questions)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["llama", "lmstudio"], default="llama")
    p.add_argument("--model", default="")
    p.add_argument("--executable", default="llama-cli")
    p.add_argument("--url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--facts", type=int, default=96)
    p.add_argument("--questions", type=int, default=24)
    p.add_argument("--relation-limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=3801)
    p.add_argument("--db", default=":memory:")
    run(p.parse_args())
