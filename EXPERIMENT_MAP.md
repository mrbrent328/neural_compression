# Neural Compression — Experiment Map

This map focuses on the actual Neural Compression research experiments, beginning with the recovered E14B branch. Preliminary teaching exercises and early demonstrations that were mainly used to learn the tools are intentionally outside this walkthrough.

Each entry records the question, what the code actually did, preserved evidence, what can safely be concluded, and what remains unknown. Code surviving is not the same as a successful experimental result.

## E14B — Whole-model latent container

- **Source:** `ROCM-test/test14b.py`
- **Related review:** [historical_reviews/test_14.md](historical_reviews/test_14.md)
- **Question:** Could a real neural-network model, stored as a GGUF file, be reorganized into compact latent representations that could later be located and reconstructed?
- **Method:** Read every tensor in a GGUF model. Large 2D weight tensors were divided into 256×256 tiles and passed through a convolutional encoder, producing 16×32×32 latent grids. Small and one-dimensional tensors were copied through unchanged. The latent data went into a binary file, while a JSON index stored tensor names, shapes, tile information, offsets, and lengths.
- **Why it was interesting:** It treated a model as many addressable local pieces, rather than one enormous undifferentiated weight file. The index made those pieces individually locatable.
- **What the apparent ratio means:** The script reports a reduction in raw float32 tensor values for encoded tiles. It is not a verified end-to-end compression ratio against the original quantized GGUF, because model costs and reconstruction fidelity were not accounted for.
- **Critical limitation:** The encoder was randomly initialized and untrained. No matching trained decoder or exact correction stream was present. The follow-up inference code could produce shape-compatible tensors for a PyTorch layer, but that did not show that the original weights or model behavior survived.
- **Evidence status:** The implementation and follow-up code are preserved. The intended tensor/hash comparison was never actually invoked, and the required reference hashes were not written into the index.
- **Conclusion:** **Foundational architecture demonstrated; compression/fidelity unproven.** This is not a failed proof that latent storage cannot work—the decisive test was never completed.
- **Possible connection worth tracking:** Local tiled representations plus a persistent address index may be an early ancestor of the later local-unit and relational-structure experiments. That is a retrospective connection, not something E14B itself proved.

## Walkthrough scope

Continue from E14B through the actual experimental branches and their revisions. Keep tightly related script revisions together where they are implementation iterations of the same question; split them when the hypothesis, control, or measured target changes. The 30-hour burst should be reconstructed as a connected sequence, not treated as dozens of unrelated files.

For each stop, we will give a short plain-English explanation, pause for the user's observations, then record the summary before advancing. We will preserve missing results as missing rather than infer them from filenames or code.


## E14B follow-up — `artifacts/test14_inference.py`

- **Question:** Can the stored latent entries be read back, expanded into tensor-shaped arrays, and used in a neural-network layer?
- **Method:** Read the JSON index, seek to a tensor's byte offset in the latent binary file, load the latent grid, pass it through a transpose-convolution decoder, stitch reconstructed tiles together, crop padding, and reshape to the original tensor dimensions. The script then copies the reconstructed weight and bias into a PyTorch `Linear` layer and runs a forward pass on random input.
- **What it demonstrates:** The storage/index path can be read, and the decoded arrays can be made shape-compatible with a PyTorch layer well enough to execute a forward pass.
- **Critical limitation:** The decoder is newly initialized in this script. It is not a trained matching decoder paired with the encoder from E14B. The test uses random input and checks only that execution completes; it does not compare reconstructed weights with the originals or compare model outputs.
- **Conclusion:** **Reconstruction plumbing / executable-shape demonstration only.** It does not establish weight fidelity, preserved model behavior, or useful compression.
- **Next file:** `artifacts/test14_inference_hashcheck.py`, which attempts to add the missing fidelity check.
