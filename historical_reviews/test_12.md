# Historical Review — Test 12

## Source
- `artifacts/test12.py`

## Purpose
Test 12 explores a different answer to the recurring model-cost problem: rather than train a model beforehand and transmit/store that learned model, initialize a model from shared starting state and update it online as the stream is processed.

The underlying question is:

**Can the model become decoder-reconstructible state that both sides learn from already decoded data, reducing or eliminating explicit model-transmission cost?**

## What changed from the preceding line
Earlier experiments repeatedly encountered the same representation economics problem:

`compressed representation = learned model/state + unexplained data`

Test 12 moves toward:

`compressed representation = unexplained data`

with

`model state = deterministic function(decoded history)`

This is a conceptual move from a static learned model toward an evolving online predictor state.

## Model
The script defines a byte-level neural predictor with:

- 16-byte context;
- embedding dimension 32;
- hidden dimension 64;
- 256 byte outputs plus a padding symbol in the embedding table;
- cross-entropy training;
- Adam optimizer with learning rate 0.002.

## Online update protocol
`compress_online_batched()` walks through the file in mini-batches of 256 bytes.

For each batch:

1. construct contexts;
2. evaluate the current model;
3. compute `-log2 p(actual byte)` for each target;
4. add those ideal information costs to the running total;
5. train the model on that batch;
6. move to the next batch with the updated model.

The returned neural size is `ceil(total_bits / 8)`.

This is therefore an online predictive coding *estimate*, not yet an emitted codec stream.

## Benchmark data
The script benchmarks on real-world data rather than generated corpora:

- a 1 MB slice of `enwik8`;
- the Requests project's `sessions.py` source file.

It compares the neural estimate with zlib/gzip and, when installed, Zstandard.

## Important methodological limitations
### 1. No actual compressed bitstream
The script does not contain a range/arithmetic encoder or archive format. It only sums ideal cross-entropy costs. Therefore `Neural (B)` is an entropy-coded size estimate, not a measured stored representation.

### 2. No decoder
There is no matching reconstruction path demonstrating that a decoder can reproduce the evolving neural state and recover the original bytes exactly.

### 3. Batched context causality is not demonstrated
The contexts are indexed directly from the complete original data tensor. Within a 256-byte batch, later positions can use earlier true bytes from the same batch while the model probabilities for the entire batch are evaluated together.

A genuinely causal decoder would need the protocol to ensure that every context symbol is already reconstructed before that probability is generated, or define a batch coding/update scheme that provides equivalent information to both sides.

### 4. Initial-state reproducibility is unresolved
The model is freshly initialized, but the script does not set a random seed or transmit initial weights. A standalone encoder/decoder pair would require identical initial state.

### 5. Numerical synchronization is unresolved
Even with shared initialization, online optimizer updates performed with GPU floating-point operations would need deterministic enough behavior for encoder and decoder to remain probability-synchronized over the stream.

## Preserved evidence
The current repository preserves the Test 12 source implementation. This review does not treat its printed neural byte estimate as evidence of a complete codec because no bitstream or decoder is implemented in the preserved script.

## Retrospective interpretation
Test 12 is an incremental implementation but an important architectural thought.

The recurring question since the earliest tests had been how to obtain useful learned structure without paying too much to communicate the learned structure itself. Test 12 proposes a different solution:

**do not transmit the model; let both sides recreate the model by learning from the same decoded history.**

That idea is a clear precursor to the later deterministic online neural-compression line, where synchronized evolving predictor states become central.

The preserved experiment does not yet prove that architecture because it estimates ideal entropy cost, lacks an actual coder/decoder, and does not fully enforce decoder-causal batching or reproducible model evolution.

## Historical transition

`static learned model + transmission/storage cost`

→

`shared initial model + online updates from stream history`

→

`potentially zero transmitted model state if encoder and decoder can evolve identically`

## Closing label
**Online self-reconstructing model-state hypothesis: replaces transmitted pretrained structure with a model learned progressively from the stream. The preserved implementation estimates entropy rather than producing a codec, and its batching/reproducibility do not yet demonstrate synchronized causal decoding.**