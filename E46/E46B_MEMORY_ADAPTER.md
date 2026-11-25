# E46-B — Activation-level persistent memory adapter

**Status:** prototype scaffold; no pretrained-model result claimed.  
**Host candidate:** Qwen3-1.7B-Base, subject to local runtime validation.  
**Base-model policy:** freeze every base-model parameter; train only the adapter interface.  
**Primary target:** a memory adapter for an already-trained network, not prompt stuffing or a replacement language model.

## Research question

Can a small external memory system preserve selected contextual internal representations from a frozen pretrained model, retrieve them in a later interaction, and reinject useful information into the model's computation through a learned interface?

The target pipeline is:

`activation → address + compressed payload → persistent store`

`current activation → query → retrieved payload → decoded residual → selected layer`

## Prototype architecture

1. **Capture point:** a selected decoder block's output residual stream.
2. **Writer:** learned address and payload projections encode a sequence-mean pooled contextual activation.
3. **Persistent bank:** CPU-resident address/payload records with metadata; saved independently of the model process.
4. **Query encoder:** maps a later activation to the address space.
5. **Address retrieval:** cosine similarity selects top-k records; address retrieval and payload interpretation are separate.
6. **Reader:** decodes the retrieved payload into the host hidden dimension.
7. **Conditioning interface:** a residual hook adds the decoded vector at the selected decoder block. A zero-initialized scalar gate makes a fresh adapter behavior-neutral until trained or calibrated.

The prototype broadcasts one decoded memory vector across the current sequence. This is a deliberately simple first conditioning boundary, not a claim that sequence-mean pooling is optimal. Token-level, multi-layer, and controller-driven retrieval are later ablations.

## Implementation boundary

Base-model parameters are frozen and not rewritten. The adapter adds runtime computation at the selected layer. During adapter training, do not wrap the entire model forward pass in torch.no_grad(): frozen base weights can still pass gradients to the adapter hook.

The hook expects a decoder block output that is either a hidden-state tensor or a tuple whose first item is the hidden-state tensor. Hugging Face module paths differ. Inspect model.named_modules() and select a real decoder block.

## Files and first local check

- e46b_memory_adapter.py — persistent bank, writer/query/reader, residual injection hook, save/load.
- ../tests/test_e46b_memory_adapter.py — CPU-only tests using a toy decoder.
- e46b_qwen_host_check.py — local checkpoint activation capture, persistent reload, and injection smoke test.

From the repository root, with PyTorch installed:

```bash
python -m unittest discover -s tests -p 'test_e46b_memory_adapter.py' -v
```

## Next Qwen host validation

A host-specific runner should load the exact local checkpoint with AutoModelForCausalLM, print likely decoder-block paths, verify hidden-state width and output structure, capture an activation, write a record, terminate, reload the adapter/bank in a fresh process, and verify a later prompt reaches the hook with finite tensors.

Compare gate-zero output with a calibrated nonzero gate. Snapshot or hash all base parameters before/after to prove that no base weights changed.

## Qwen host smoke test

After the checkpoint is downloaded locally, run the capture phase:

```bash
python E46/e46b_qwen_host_check.py capture --model-path /absolute/path/to/Qwen3-1.7B-Base
```

Then run injection in a separate process:

```bash
python E46/e46b_qwen_host_check.py inject --model-path /absolute/path/to/Qwen3-1.7B-Base --prompt "A different question in a later session." --gate 0.25
```

The capture command auto-selects a likely middle decoder block; pass --layer-path with an exact path if needed. Use a meaningfully different prompt during injection: repeating the capture prompt only checks the wiring, not cross-context reuse. The runner rejects a layer mismatch because activations from different layer locations are not interchangeable. The inject command reloads the adapter and memory bank from disk, runs a baseline and hooked forward pass, and reports the logit delta. A nonzero delta verifies that the conditioning hook is active, not that memory improves task performance. Use --gate 0 to check the behavior-neutral path.

The script requires transformers and PyTorch and uses local_files_only=True. It does not download model files or transmit prompts.

## Training and evaluation are not yet implemented

The writer, query encoder, and reader are initialized modules, not trained components. The zero gate means the fresh adapter does not change model output. A nonzero gate proves the injection path runs; it does not prove retrieved memory is useful.

The next learning experiment must train only the adapter against a controlled objective and compare no memory, raw activations, learned compressed payloads at multiple widths, shuffled/mismatched records, text memory, and a disabled adapter.

Measure task accuracy, unsupported-answer rate, activation reconstruction, total storage including adapter and metadata, retrieval latency, end-to-end latency, and peak RAM/VRAM. Use distinct write/read contexts so evaluation tests cross-session reuse.

## E45 connection and limits

E45 Phase 3C found perfect address retrieval in its synthetic setup, while unseen-value reconstruction depended strongly on payload width. Carry 8D/16D/24D/32D as explicit quality/storage ablations when training the activation codec. Those dimensions do not automatically transfer to a 2048-wide transformer hidden state.

No E46-B result establishes useful pretrained-model memory, semantic generalization, continual learning, computational savings, or cross-model portability.

## Planned sequence

- **E46-B1 — activation access:** inspect Qwen module structure and verify capture/injection.
- **E46-B2 — persistence:** save/reload adapter and memory across fresh processes.
- **E46-C1 — learned interface:** train writer/query/reader while base weights stay frozen.
- **E46-C2 — memory utility:** compare raw, compressed, shuffled, text, and no-memory controls.
- **E46-C3 — selective retrieval:** train a controller to retrieve only when expected reuse benefit justifies cost.
- **E46-D — portability:** test a second host model only after same-model integration succeeds.
