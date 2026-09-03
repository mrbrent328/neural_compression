#!/usr/bin/env python3
"""Local inference backends for E38.

The experiment uses an interchangeable backend so LM Studio and llama.cpp
can be compared without changing the memory/evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import time
import urllib.request


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LlamaCppBackend:
    """Invoke llama.cpp's llama-cli executable."""
    def __init__(self, model: str, executable: str = "llama-cli", n_ctx: int = 4096, n_predict: int = 256) -> None:
        self.model = model
        self.executable = executable
        self.n_ctx = n_ctx
        self.n_predict = n_predict

    def generate(self, prompt: str) -> Generation:
        cmd = [self.executable, "-m", self.model, "-c", str(self.n_ctx), "-n", str(self.n_predict), "-p", prompt, "--no-display-prompt"]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return Generation(proc.stdout.strip(), time.perf_counter() - start)


class LMStudioBackend:
    """Use LM Studio's OpenAI-compatible local server."""
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "local-model", max_tokens: int = 256) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Generation:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as response:
            data = json.load(response)
        latency = time.perf_counter() - start
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return Generation(
            choice["message"]["content"].strip(),
            latency,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
