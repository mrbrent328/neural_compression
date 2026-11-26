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

For each stop, we will give a short plain-English explanation, pause for the user's observations, then record the summary before advancing. We will preserve missing results as missing rather than infer them from code.

## E14B follow-up — `artifacts/test14_inference.py`

- **Question:** Can the stored latent entries be read back, expanded into tensor-shaped arrays, and used in a neural-network layer?
- **Method:** Read the JSON index, seek to a tensor's byte offset in the latent binary file, load the latent grid, pass it through a transpose-convolution decoder, stitch reconstructed tiles together, crop padding, and reshape to the original tensor dimensions. The script then copies the reconstructed weight and bias into a PyTorch `Linear` layer and runs a forward pass on random input.
- **What it demonstrates:** The storage/index path can be read, and the decoded arrays can be made shape-compatible with a PyTorch layer well enough to execute a forward pass.
- **Critical limitation:** The decoder is newly initialized in this script. It is not a trained matching decoder paired with the encoder from E14B. The test uses random input and checks only that execution completes; it does not compare reconstructed weights with the originals or compare model outputs.
- **Conclusion:** **Reconstruction plumbing / executable-shape demonstration only.** It does not establish weight fidelity, preserved model behavior, or useful compression.

## E14B fidelity-check follow-up — `artifacts/test14_inference_hashcheck.py`

- **Question:** Can reconstruction be checked against the original tensor, rather than merely checking that a layer runs?
- **Added checks:** Defines `tensor_sha256()` and `compare_tensors()`. The latter would compare hashes, exact tensor equality, count exact elements, and report maximum and mean absolute error.
- **What the script actually runs:** It reconstructs the selected weight and bias, computes and prints their hashes, and looks for optional reference hashes (`sha256`, `hash`, or `sha256_hash`) in the JSON index. It then runs the same forward-pass sanity test.
- **Critical limitation:** `compare_tensors()` is defined but never called. The encoder/index path does not provide reference hashes in the reviewed implementation, so the script normally has no original hash to compare against. Printing a hash of the reconstruction alone does not establish fidelity. It also does not compare original and reconstructed layer outputs.
- **Conclusion:** **A proposed fidelity-testing mechanism, not a completed fidelity test.** The E14 branch still does not show that the original weights or model behavior were preserved.
- **Branch-level takeaway:** E14 progressed from encoding tensors, to packaging a whole model, to attempting reconstruction and adding integrity-check code. But the essential experiment—train a paired encoder/decoder and measure reconstruction error against the originals—was not completed in these files.

## E14A — Single-tensor latent prototype (`artifacts/test14a.py`)

- **Question:** Could one large GGUF weight matrix be divided into local tiles and represented by a smaller latent tensor?
- **Method:** Selects the first GGUF tensor that is 2D and has at least 256 rows, converts its data to float32, divides it into complete 256×256 tiles, and passes batches through a convolutional autoencoder. The encoder maps each tile to 16 channels at 1/8 the height and width. The script concatenates and times the latent outputs.
- **What it demonstrates:** A single real-model tensor can be loaded, tiled, passed through an encoder, and measured for latent tensor size and runtime.
- **Critical limitation:** The autoencoder is randomly initialized and only set to evaluation mode; there is no training. The decoder is defined but its reconstruction is discarded. No reconstruction error, original-versus-reconstructed comparison, or model behavior test is performed. Only complete tiles are included, so edge remainders are omitted.
- **Compression caveat:** The printed ratio compares the float32 tile tensor with the float32 latent tensor. It is a raw element/byte-count ratio for this representation, not a verified end-to-end compressed GGUF size or a fidelity-preserving compression result.
- **Conclusion:** **Single-tensor tiling and latent-size/timing prototype.** It establishes the first stage of the pipeline, not successful weight compression.
- **Relationship to E14B:** E14B scales this basic idea from one selected tensor to a whole-model container with an index. The central missing requirement in both is a trained encoder/decoder pair and measured reconstruction fidelity.

## Near-duplicate — `artifacts/testGGUF.py`

- **Relationship:** The reviewed source is effectively a copy of E14A's single-tensor tiling and latent-size/timing prototype, with a placeholder model path. It uses the same untrained autoencoder and does not add a new reconstruction or fidelity test.
- **Conclusion:** **Implementation copy / rerun variant, not a distinct research result.** Keep it in the archive, but do not count it as an independent experiment.

## Experiment 7 — Trained latent weight recovery

- **Source:** `artifacts/Compression_test_190826/experiment7_latent_weight_recovery.py`
- **Question:** If real model weights are compressed into a learned latent representation, can the decoder recover those weights—especially tiles it was not trained on?
- **What changed:** Unlike E14A/E14B, this autoencoder is actually trained. It selects one real floating-point GGUF tensor, converts it to a two-dimensional matrix, pads and divides it into 64×64 tiles, then trains separate autoencoders with latent sizes 4, 8, 16, and 32.
- **How it tests:** It holds out 20% of the tiles, trains on the other 80%, and measures reconstruction error on both groups. It reports average error, worst error, and correlation. It also calculates the raw latent-size reduction and the size of the shared decoder. Finally, it retrains the best latent-size model and reconstructs the full selected tensor.
- **Why this is a major step:** This is the first file in this branch that directly trains the encoder and decoder to preserve real model-weight information, then checks whether the learned representation works on tiles withheld from training.
- **Important limits:** It tests one selected tensor, not a whole model. Its tile split is random, so neighboring or structurally similar tiles may appear in both training and test sets. Its reported latent ratio counts float32 latent values but does not include the shared decoder, metadata, or other storage overhead. The script's output/results are not present in this source file, so code alone cannot tell us which latent size performed best or how good reconstruction actually was.
- **Conclusion:** **A real learned reconstruction experiment with a held-out test design.** Its numerical outcome must be recovered from run logs or result files before drawing a conclusion. Even good weight reconstruction would not by itself prove that model outputs or language behavior are preserved.
