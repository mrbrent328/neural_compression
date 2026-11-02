# Aetheria Video — Research Architecture

**Status:** Proposed architecture / pre-experiment  
**Branch:** `research/aetheria-video-architecture`  
**Research mode:** Hybrid neural + conventional coding  
**Optimization target:** Balanced storage efficiency, visual quality, and real-time decoding  
**First target:** Short 720p/1080p natural-video clips; 4K is a later scaling test

## 1. Research question

Can a hybrid codec reduce the total information needed to reconstruct a video by combining:

1. reusable visual structure shared across many observations;
2. temporal prediction over previously reconstructed content;
3. compact quantized latent representations;
4. explicitly coded residual information; and
5. deterministic entropy decoding?

The specific Aetheria hypothesis is that **persistent reusable visual structure can reduce the rate needed to describe later observations beyond what short-range frame prediction alone achieves**, while keeping decoder cost practical.

This is a hypothesis to test, not an established result. Neural video compression, learned temporal prediction, latent residual coding, and hybrid codecs are already active research areas. Aetheria must establish its contribution through controlled ablations and complete rate/quality/complexity accounting.

## 2. Design principles inherited from Persistent Structure

### E40 — Adaptive Residual Codec

Reuse the engineering principles, not the byte-level predictor as the video model:

- predictor and decoder state must remain synchronized;
- only the selected coding path updates adaptive state;
- representation choice must use measured complete bit cost, including metadata;
- residuals must be explicitly decodable;
- malformed, truncated, and incompatible streams must fail safely;
- the codec must be a real file-oriented encoder/decoder, not an offline compression estimate.

E40's MP4 result increased file size slightly. This is expected for a weak byte predictor applied to already-compressed data and is a reason to move prediction into a learned visual/temporal representation, not evidence that the adaptive residual mechanism itself is invalid.

### E41 — Persistent Relational Memory

Reuse the hypothesis that relationships between observations can carry predictive information not available in an isolated observation.

For video, memory may represent recurring visual features, persistent entities, scene structure, and relationships. Initially, do not assume that semantic object identities can be recovered reliably. Start with learned content-addressed slots and explicit ablations.

### Shared latent constructors

Reuse the lesson that tiny per-instance codes are only meaningful when the shared constructor cost is counted and amortized across enough instances. Always report:

- shared model/profile bytes;
- per-video dictionary or memory bytes;
- per-GOP and per-frame coded bytes;
- total bytes for one video;
- amortized total across a library of videos.

Never describe a per-video latent as "free" because its decoder is shared.

## 3. Proposed hybrid architecture

The encoder may perform expensive full-sequence analysis and lookahead. The decoder must use only the model/profile and information carried by the bitstream or deterministically derived from previously decoded content.

### 3.1 Shared codec profile

A frozen, versioned model profile contains the reusable learned machinery:

- spatial analysis transform;
- spatial synthesis transform;
- temporal predictor;
- persistent-memory update/read functions;
- latent entropy model;
- quantization parameters and normative inference configuration.

The profile is identified by a stable ID and cryptographic hash. The bitstream must identify the exact profile required for decoding. A decoder must reject an unavailable or incompatible profile rather than silently substituting another model.

Model weights are not transmitted per video when the recipient already has the exact profile. Their storage and distribution cost must nevertheless be reported and amortized explicitly.

### 3.2 Sequence/GOP analysis and persistent memory

The encoder analyzes a sequence or GOP and discovers reusable structure. A sequence-level memory may contain:

- learned visual feature prototypes;
- content-addressable feature slots;
- relationships between recurring features;
- scene-level context useful for later prediction.

Memory has two legal sources:

1. **Derived state:** computed deterministically by the decoder from already reconstructed frames and previously decoded state.
2. **Coded state:** generated or optimized by the encoder but explicitly quantized and transmitted in the bitstream.

No source-derived memory may influence reconstruction without its information being represented in the bitstream. This is a critical information-accounting rule.

Memory updates are versioned and bounded. GOP boundaries provide reset points so errors do not propagate indefinitely and random access remains possible.

### 3.3 Spatial analysis transform

For frame x_t, a learned analysis transform produces a compact latent y_t:

`y_t = g_a(x_t)`

The first implementation should use a modest convolutional encoder/decoder and fixed tensor shapes. Do not begin with full 3D reconstruction, actor-specific meshes, NeRFs, or a large generative model.

### 3.4 Temporal prediction in latent space

Predict the current latent from previously reconstructed latents and persistent memory:

`p_t = P( y_hat_(t-1), ..., y_hat_(t-k), M_t )`

The encoder computes the latent residual:

`r_t = y_t - p_t`

The residual is quantized:

`q_t = Q(r_t; step_t)`

The decoder recreates the latent using the same prediction and quantized residual:

`y_hat_t = p_t + Q_inverse(q_t; step_t)`

The synthesis transform produces the reconstructed frame:

`x_hat_t = g_s(y_hat_t)`

All reference inputs to the decoder predictor must be reconstructed values, not original source frames. This prevents encoder/decoder drift.

### 3.5 Persistent visual memory

Memory is updated from reconstructed or explicitly coded information, not hidden source information:

`M_(t+1) = U(M_t, y_hat_t, q_t, metadata_t)`

Memory should be content-addressable and relational, but the first experiment should keep it small and inspectable. Compare at least:

- no persistent memory;
- simple pooled temporal state;
- content-addressed slots;
- content-addressed slots plus relations.

This directly tests whether persistent structure contributes beyond a larger temporal context window.

### 3.6 Entropy coding and bitstream

Quantized latents, memory updates, keyframe payloads, and required metadata are entropy-coded. E40's synchronized-state principle can inform the entropy model, but the first prototype should prioritize a simple correct bitstream over a novel entropy coder.

The bitstream must carry or unambiguously define:

- magic and format version;
- codec profile ID/hash;
- dimensions, frame rate, pixel format, colour metadata;
- GOP/keyframe structure;
- quantization and rate-control parameters;
- coded memory initialization and updates;
- keyframe and latent payload lengths;
- entropy-coder state/reset boundaries;
- checksums and stream termination.

Audio is out of scope for the first video experiments. A later container can mux audio and subtitles without coupling them to the video codec.

## 4. Decoder requirements

The decoder is a first-class design constraint, not a later optimization.

- Deterministic reconstruction for a given profile and bitstream.
- No source-video access, encoder lookahead, or online training.
- Bounded reference-frame and memory buffers.
- GOP reset and random-access support.
- Streaming decode without loading the entire file.
- Explicit handling of missing profiles, corrupt packets, truncation, and unsupported versions.
- Measure total decoder wall time, memory traffic, model calls, and peak memory—not only neural FLOPs.
- Consider integerized inference after the floating-point prototype is validated.

The target is balanced real-time playback, initially at 1080p/30 fps on a documented reference machine. This is a target, not a current capability claim. 4K/30 fps is a later milestone.

## 5. Rate-distortion objective

The encoder should optimize a rate-distortion objective:

`L = R_total + lambda * D(x, x_hat)`

where:

- `R_total` is the actual or accurately estimated number of bits, including headers, memory, model/profile allocation, and entropy-coded payload;
- `D` is a distortion measure;
- `lambda` controls the quality/bitrate trade-off.

Use a combination of pixel and perceptual metrics during development. Report PSNR and MS-SSIM; add VMAF for perceptual comparison where available. Do not optimize only a learned perceptual metric, because it can hide detail loss or temporal artifacts.

The encoder may use full-sequence lookahead, but every decision that changes decoder reconstruction must be represented in the bitstream.

## 6. Experimental controls

Every Aetheria result must be compared against:

1. **Conventional baseline:** a pinned x265/HEVC configuration.
2. **Modern conventional baseline:** a pinned AV1 encoder configuration.
3. **Temporal baseline:** the same learned spatial codec and temporal predictor, but with persistent memory disabled.
4. **Aetheria memory variants:** pooled state, content-addressed slots, and slots plus relations.

Compare at matched visual quality as well as matched bitrate. A comparison at different quality levels is not a valid compression win.

Record exact encoder versions, command lines, model hashes, datasets, frame ranges, resolution, frame rate, chroma format, bit depth, and all rate-control settings.

## 7. Evaluation metrics

### Compression and quality

- total bitstream bytes and bits per pixel;
- bitrate at matched PSNR / MS-SSIM / VMAF;
- rate-distortion curves across several quality settings;
- per-GOP and per-frame bit allocation;
- shared profile bytes and amortized bytes per video;
- memory/dictionary overhead and metadata overhead.

### Playback and resource cost

- encode FPS and total encoding time;
- decode FPS and first-frame latency;
- peak RAM and VRAM;
- CPU/GPU utilization;
- model calls and memory transfers per frame;
- random-access latency after seeking;
- error propagation after a damaged GOP.

### Correctness

- deterministic decode tests;
- repeated encode/decode tests;
- checksum of reconstructed output for deterministic runs;
- malformed/truncated stream tests;
- profile mismatch tests;
- encoder/decoder state synchronization tests.

## 8. Experiment sequence

### AV0 — Benchmark and bitstream foundation

**Question:** Can we build a reproducible video benchmark harness before introducing learned compression?

- Select a small set of short, diverse clips.
- Normalize to a documented resolution, frame rate, colour format, and bit depth.
- Produce pinned x265 and AV1 baselines at several quality points.
- Store source hashes, commands, output sizes, and quality metrics.
- Implement a minimal versioned container/packet reader and writer.
- Add streaming decode and corruption/truncation tests.

**Exit condition:** A repeatable baseline report and a tested container skeleton. No compression claim.

### AV1 — Learned temporal latent baseline

**Question:** Can a learned spatial transform plus causal latent prediction produce a valid compact video bitstream?

- Train a small spatial analysis/synthesis transform.
- Use periodic intra/keyframes and causal prediction between them.
- Quantize latent residuals and entropy-code them.
- Decoder predicts from previously reconstructed latents only.
- Persistent relational memory is disabled.

**Exit condition:** End-to-end encode/decode, stable reconstruction, measured rate-distortion curve, and measured decode speed.

### AV2 — Persistent memory ablation

**Question:** Does persistent reusable structure reduce total coded bits beyond the AV1 temporal baseline?

Compare:
- no memory;
- pooled temporal memory;
- content-addressed feature slots;
- feature slots plus relational links.

Hold the spatial transform, training data, compute budget, quality target, and entropy coder constant. Count all coded memory updates and model costs.

**Primary success criterion:** At matched quality, persistent memory reduces total bitrate relative to the no-memory temporal baseline without violating the decode target.

### AV3 — Long-range reuse and multi-video amortization

**Question:** Can a shared visual dictionary or profile learned across multiple videos reduce per-video storage enough to offset its model cost?

Test:
- one-video profile;
- multi-video shared profile;
- profile plus per-video dictionary;
- profile reuse on held-out videos.

Report break-even library size and generalization to unseen content. Avoid leakage between training and evaluation clips.

### AV4 — Runtime and resolution scaling

Only after AV2 demonstrates a real rate-distortion benefit:
- optimize inference and memory layout;
- investigate integerized inference;
- measure 1080p/30 fps;
- then test 4K and longer sequences.

## 9. Initial dataset strategy

Start with short controlled clips that expose distinct failure modes:

- static camera with moving subject;
- camera pan across a mostly static scene;
- repeated objects/characters;
- occlusion and reappearance;
- lighting changes;
- fine textures and foliage;
- fast motion and scene cuts.

Use synthetic clips first for controlled object identity and motion tests, then public natural-video benchmarks for external validity. Keep test clips held out from training and profile fitting.

Do not begin with a five-season television corpus. The first milestone is a small, interpretable experiment.

## 10. Implementation layout (proposed)

```text
aetheria_video/
  README.md
  configs/
    av0_benchmark.yaml
    av1_latent_baseline.yaml
    av2_memory_ablation.yaml
  aetheria_video/
    bitstream/
      format.py
      reader.py
      writer.py
    codec/
      encoder.py
      decoder.py
      gop.py
      rate_control.py
    models/
      analysis.py
      synthesis.py
      temporal_predictor.py
      persistent_memory.py
      entropy_model.py
    metrics/
      quality.py
      rate.py
      runtime.py
  experiments/
    AV0_benchmark/
    AV1_temporal_latent/
    AV2_persistent_memory/
  tests/
    test_bitstream.py
    test_roundtrip.py
    test_state_sync.py
    test_corruption.py
    test_streaming_decode.py
  results/
    README.md
```

This is a proposed layout. Do not move or rewrite the historical E-series experiments. Aetheria Video should be a separate, traceable research branch.

## 11. Risks and explicit non-goals

- **Information displacement:** a small latent may hide information in a large decoder or per-video dictionary. Count everything.
- **Semantic hallucination:** a generative decoder may invent plausible detail. Residuals and distortion measurements must expose this.
- **Temporal drift:** prediction from mismatched encoder/source and decoder/reconstruction states causes accumulating errors. Always predict from reconstructed references.
- **Memory leakage:** encoder-only source information cannot silently enter decoder state.
- **Overfitting:** a scene-specific representation may compress one clip exceptionally while failing on new clips. Include held-out content.
- **Complexity trap:** do not begin with 3D meshes, NeRFs, Gaussian splats, actor tracking, or semantic segmentation. They are possible later representations, not prerequisites.
- **Baseline fairness:** compare at matched quality and include model, memory, metadata, and container overhead.
- **Novelty discipline:** neural video compression is an established research field. Aetheria's claimed contribution must be supported by the persistent-memory ablation, not by using neural networks in a video codec.

## 12. First decision gate

Do not train the persistent-memory model first.

Complete AV0, then build AV1 as the control. AV2 is the decisive experiment. If AV2 does not beat the no-memory baseline at matched quality and practical decode cost, investigate why before adding more architectural complexity.

The central research question remains:

> Does a compact persistent representation of reusable visual structure reduce the total information required to reconstruct future observations, beyond ordinary temporal prediction?

## References and starting points

- Jia et al., **Emerging Advances in Learned Video Compression: Models, Systems and Beyond**, IJCAI 2025. Survey of learned video compression architectures, temporal prediction, systems, and evaluation.
- Jia et al., **Towards Practical Real-Time Neural Video Compression**, CVPR 2025. Practical decoder/runtime design; emphasizes operational overhead, memory I/O, and function-call costs as well as compute.
- Existing project foundations: `E40_adaptive_residual_codec.py`, `test_E40_adaptive_residual_codec.py`, `E41_relational_concept_memory.py`, `NEURAL_COMPRESSION_AUDIT.md`, and `RESEARCH_LOG.md`.

---

**Current classification:** PROPOSED / NOT YET TESTED  
**Next action:** Implement AV0 benchmark harness and pin baseline configurations.
