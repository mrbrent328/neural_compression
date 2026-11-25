#!/usr/bin/env python3
"""E46 Gate 1: smoke-test a local LM Studio OpenAI-compatible server.

No third-party Python packages are required.
Default server URL: http://localhost:1234/v1

Examples:
  python E46/lmstudio_smoke.py
  python E46/lmstudio_smoke.py --model openai/gpt-oss-20b
  python E46/lmstudio_smoke.py --base-url http://127.0.0.1:1234/v1 --timeout 300
  python E46/lmstudio_smoke.py --api-key "$LM_API_TOKEN"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, *, method: str = "GET", payload=None,
                 api_key: str | None = None, timeout: int = 120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach {url}: {exc.reason}. "
            "Start the LM Studio local server and verify its host/port."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Request timed out after {timeout}s: {url}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL (default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="Exact LM Studio model ID; auto-selects if omitted")
    parser.add_argument("--api-key", default=os.environ.get("LM_API_TOKEN") or None,
                        help="Optional API token; defaults to LM_API_TOKEN")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/E46/lmstudio_smoke.json"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    started = time.perf_counter()
    models_doc, _ = request_json(
        f"{base}/models", api_key=args.api_key, timeout=args.timeout
    )
    model_rows = models_doc.get("data", []) if isinstance(models_doc, dict) else []
    model_ids = [row.get("id") for row in model_rows if row.get("id")]
    if not model_ids:
        raise RuntimeError(
            "The server responded to /models but returned no model IDs. "
            "Load a chat model in LM Studio, then retry."
        )

    if args.model:
        model_id = args.model
        if model_id not in model_ids:
            raise RuntimeError(
                f"Requested model '{model_id}' is not listed by this server. "
                f"Available IDs: {', '.join(model_ids)}"
            )
    else:
        model_id = next(
            (m for m in model_ids if "gpt-oss-20b" in m.lower()),
            model_ids[0],
        )

    prompt = (
        "Reply with exactly: E46 LM Studio connection successful."
    )
    chat_started = time.perf_counter()
    chat_doc, _ = request_json(
        f"{base}/chat/completions",
        method="POST",
        payload={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 48,
            "stream": False,
        },
        api_key=args.api_key,
        timeout=args.timeout,
    )
    chat_seconds = time.perf_counter() - chat_started

    choices = chat_doc.get("choices", [])
    if not choices:
        raise RuntimeError(
            "Chat endpoint returned no choices: " +
            json.dumps(chat_doc, ensure_ascii=False)[:1200]
        )
    message = choices[0].get("message", {})
    answer = message.get("content", "")
    if isinstance(answer, list):
        answer = json.dumps(answer, ensure_ascii=False)

    result = {
        "experiment": "E46_LM_Studio_connection_smoke",
        "status": "PASS",
        "base_url": base,
        "model_id": model_id,
        "models_visible": model_ids,
        "prompt": prompt,
        "response": answer,
        "chat_latency_seconds": round(chat_seconds, 4),
        "total_latency_seconds": round(time.perf_counter() - started, 4),
        "usage": chat_doc.get("usage"),
        "finish_reason": choices[0].get("finish_reason"),
        "note": (
            "Connection and text generation only. This does not test memory, "
            "embeddings, hidden-state access, or model adaptation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"E46 smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
