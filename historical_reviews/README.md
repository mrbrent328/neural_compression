# Neural Compression — Historical Experiment Reviews

This directory is a retrospective reconstruction of the early Neural Compression experiment sequence.

The goal is to review each historical experiment once, preserve what it actually tested, distinguish preserved evidence from hindsight interpretation, and build a durable account of how the project developed without repeatedly re-summarizing old experiments from conversation history.

## Method
Each experiment receives its own review file. Reviews should preserve the same basic structure:

- original source/script;
- historical position when known;
- purpose/question;
- experimental design;
- preserved result/evidence;
- limitations or accounting issues;
- retrospective interpretation using later project knowledge;
- what the experiment motivated next.

Claims about historical numerical outcomes should only be made when an authoritative result/log/output is preserved. When only code survives, the review should distinguish what the experiment was designed to test from what the original run actually demonstrated.

## Reconstructed early sequence

### Test C
Review: [`test_c.md`](test_c.md)

Early learned text-prediction/residual experiment. Established that compression must account for the cost of the predictor/model as well as the residual. The adaptive context model highlighted the advantage of state that can be reconstructed causally without transmission.

**Transition:** learned prediction → complete representation accounting.

### Test D
Review: [`test_d.md`](test_d.md)

Converted the learned predictor into a proper 256-symbol probability model feeding an arithmetic coder. Added categorical cross-entropy training, held-out evaluation, causal exact decoding, model-size accounting, and SHA-256 reconstruction verification. Performance comparisons have important shared-prefix/control accounting limitations.

**Transition:** predicted value + residual → learned probability distribution + entropy coding.

### Test 2
Review: [`test_2.md`](test_2.md)

Moved from temporal next-symbol prediction toward learned representation of whole 1,024-byte regions. A 16-dimensional latent bottleneck attempts to explain block structure while an exact modulo-256 residual preserves every missed detail. The experiment explicitly measures residual compressibility and entropy and attempts complete model + residual + metadata accounting.

The preserved implementation is not yet a standalone codec: reconstruction uses in-memory predictions, block-specific latent vectors are not transmitted or otherwise regenerable by the decoder, and package framing lacks an explicit model/residual boundary.

**Transition:** learned prediction → learned structural representation + exact correction. Exposes the problem of how useful latent state can itself be represented or regenerated cheaply.

### Test 3
Review: [`test_3.md`](test_3.md)

Replaced Test 2's source-dependent block latents with causal prediction state derived from byte history available to the decoder. Introduced a 75/25 train/held-out design across structured text, source-code-like data, and a compactly generated mathematical signal. Compared previous-byte, context, neural, and simple hybrid predictors using exact residuals, model accounting, entropy measurements, exact reconstruction, and SHA-256 verification.

Test 3 substantially improves the controls and addresses Test 2's latent-communication problem, but its printed storage comparison still mixes domains: predictor totals describe the held-out 25%, while the zlib baseline describes the entire dataset.

**Transition:** transmitted local latent state → shared learned structure + decoder-reconstructible state + exact correction. Introduces broader structural generalization and primitive predictor combination.

### Test 4
Review: [`test_4.md`](test_4.md)

Tests learned next-byte predictors with multiple context widths (4, 8, 16, and 32 bytes) and explicitly chooses the model with the smallest combined compressed model + exact residual + metadata representation rather than the lowest prediction loss. This makes context/model capacity itself an economic compression variable.

The experiment still compares a learned representation of only the held-out final 25% against full-file original/zlib sizes, so whole-file accounting remains unresolved. It also verifies residual arithmetic rather than exercising an independently packaged sequential decoder.

**Transition:** decoder-reconstructible learned context → optimize how much contextual/model capacity is worth paying for. Early precursor to adaptive capacity allocation and local strategy selection.

### Test 5
Review: [`test_5.md`](test_5.md)

Returns to direct entropy coding: an 8-byte-context neural predictor generates 256-symbol probabilities that are deterministically converted to arithmetic-coder frequency tables. Unlike Tests 2–4, the held-out continuation is actually decoded sequentially from the arithmetic stream, with each decoded symbol feeding the next neural context. Exact full-sequence equality and SHA-256 verification are performed.

The arithmetic payload only represents the final 25%, while the first 75% is supplied to the decoder as an uncharged prefix and the comparison baseline remains full-file zlib. Thus the causal codec mechanism is substantially validated, but complete whole-file accounting remains unresolved.

**Transition:** optimize learned context/model cost → use learned uncertainty directly as a causal lossless arithmetic-coded stream.

### Test 6 / Latent Structure Discovery Branch
Review: [`test_6_discovery_branch.md`](test_6_discovery_branch.md)

Test 6 begins a deliberate side branch from the latent-representation idea: if useful latent structure can be stored, can the system discover that structure itself? The initial experiment searches for a compact mathematical law behind noisy observations without receiving the hidden generating equation.

The branch continues through `discover_law2.py`, `discover_law3.py`, and `blind_law_suite.py`, progressively adding irrelevant-variable rejection, multivariable interactions, explicit complexity-aware scoring, and tests across multiple hidden law families. These are symbolic-regression experiments over supplied mathematical vocabularies, not unrestricted or neural scientific discovery.

**Transition:** learn/store latent structure → discover latent structure → prefer compact explanatory/generative structure → test structural discovery across law families.

### Test 7
Review: [`test_7.md`](test_7.md)

Introduces an explicit scaling/amortization hypothesis: a learned structural representation may be too expensive on small data but become economical when its fixed cost is reused across enough information. The script compares previous-byte, context, neural, and hybrid predictive residual methods across 100 KB to 5 MB synthetic datasets while accounting for residual, model, and metadata costs.

The hypothesis is historically important, but the preserved break-even experiment is not valid evidence of the claimed curve. Smaller curve points leak into the fixed model's training region, learned totals encode only each file's final 25% while zlib encodes the full file, and the hybrid uses the true target byte to choose the better predictor.

**Transition:** model overhead hurts → ask whether reusable structural overhead can be amortized over scale → treat compression as representation economics rather than prediction quality alone.

### Test 8
Review: [`test_8.md`](test_8.md)

Repairs much of Test 7's scaling methodology and moves the experiment to real source-code/text corpora. Each scaling point trains its own model using only its first 75%, the final 25% is genuinely held out, file boundaries are respected, and the hybrid selector is learned from training-region behavior instead of consulting the held-out target byte. All predictor types are reconstructed sequentially and checked by byte equality and SHA-256.

The major accounting problem remains: learned totals cover only the held-out 25% residual plus model/metadata while the zlib baseline covers the full corpus. Neural/context model sizes are also representation estimates, and hybrid selector-table storage is not separately charged.

**Transition:** flawed synthetic scaling curve → real-corpus, leakage-free, decoder-valid scaling experiment while retaining the unresolved equal-domain accounting requirement.

### Test 9
Review: [`test_9.md`](test_9.md)

Primarily an engineering/performance iteration of Test 8. It preserves the real-corpus, independent 75/25 split, decoder-valid hybrid selection, exact reconstruction, and per-scaling-point training while moving neural training, held-out prediction, and hybrid validation onto AMD ROCm hardware with controlled batching, synchronization, throughput reporting, and GPU-memory tracking.

The code deliberately keeps exact predictive decoding sequential because each next prediction depends on previously reconstructed bytes. The underlying whole-file accounting problem remains unresolved.

**Transition:** corrected real-corpus experiment → operational GPU scaling while respecting the sequential causal decoder constraint.

### Test 10
Review: [`test_10.md`](test_10.md)

Returns to direct probability coding with the matured real-corpus methodology. Instead of collapsing the neural output to one argmax byte and compressing a residual, the full 256-way probability distribution is deterministically quantized to integer frequencies and fed directly to a sequential range coder. Encoder and decoder maintain matching causal reconstruction state, and theoretical cross-entropy is compared with actual coded payload size.

The experiment also mixes neural and context distributions directly in probability space rather than selecting a hard predictor. Exact held-out decoding is verified. The longstanding whole-file accounting mismatch remains: learned totals describe the held-out 25% plus model/metadata while the zlib baseline describes the full corpus.

**Transition:** operational hard prediction at scale → direct learned uncertainty + causal entropy coding; hard predictor combination → probability-space mixture.

### Test 11
Not currently recovered. The retrospective intentionally leaves this slot unresolved rather than inferring that no Test 11 existed.

### Test 12
Review: [`test_12.md`](test_12.md)

Introduces an online self-reconstructing model-state hypothesis. Instead of training a static model beforehand and paying to communicate that state, a fresh neural predictor is evaluated on each batch and then updated from that batch so learned structure accumulates progressively from the stream itself.

The preserved script only sums ideal cross-entropy and does not emit or decode a compressed bitstream. Its batched contexts are drawn from the full source tensor, and shared initialization / deterministic numerical synchronization are not established, so it does not yet demonstrate a synchronized causal codec.

**Transition:** transmitted static model state → shared initial state + online learning from decoded history → potentially decoder-reconstructible learned state with no explicit model payload.

### Test 13 / Latent-Native Computation Branch
Review: [`test_13.md`](test_13.md)

A four-part A→B→C→D branch exploring whether a learned latent representation can become a computational substrate rather than merely a compressed storage format. A and B create and steady-state benchmark a convolutional image encoder producing `[16, 32, 32]` spatial latents from 256×256 RGB inputs. C and D freeze that encoder and train/benchmark a downstream classifier that operates directly on the latent tensor without reconstructing the source image.

Because the preserved scripts use random dummy images, random labels, and an untrained encoder, they do not establish useful semantic preservation or classification quality. Their contribution is architectural and operational: they demonstrate and benchmark the direct latent-computation path.

**Transition:** latent representation as compressed storage → latent representation as computational state → apply latent representation to real neural-network model-weight tensors in Test 14.

### Test 14 / Model-as-Latent-State Branch
Review: [`test_14.md`](test_14.md)

Applies the latent representation idea to real neural-network model weights. The branch progresses from a single GGUF tensor tiled into convolutional latents, to a whole-model `.latents` binary container with JSON indexing, to reconstruction of latent-stored tensors and injection into an executable PyTorch layer.

The branch then attempts to add exact reconstruction/fidelity validation. The preserved hash-check script contains a meaningful original-versus-reconstructed comparison routine, but that routine is never called, while the alternate hash path expects reference hashes that the converter did not store. This implementation mistake was not noticed at the time, and the line of inquiry was discontinued before the intended fidelity test actually ran.

Accordingly, Test 14 is not evidence that latent model-weight storage failed. It establishes the architectural container/reconstruction path but leaves preservation of the original weights and model function unresolved.

**Transition:** latent-native computation → model parameters as latent-representable state → whole-model indexed latent container → need for actual reconstruction/fidelity validation.

## Historical context
The earliest work was not originally expected to become a long experiment series, so initial experiments were lettered rather than organized as a mature numbered research program. Tests C and D precede numbered Test 2. The first roughly fourteen tests were developed in a very compressed early research period, making this retrospective documentation particularly useful for reconstructing the actual conceptual progression.

Test 6 temporarily branches away from direct codec development, but the branch follows directly from the compression hypothesis: perhaps the cheapest representation of observations is the compact relationship or process that generated them.

Test 11 is currently unresolved in the preserved repository and is intentionally left as a gap pending future recovery.

The repository also contains a later root-level `experiment10`→`experiment16` lineage whose numbering collides with the early `artifacts/test*` sequence. Those files are treated as a separate historical sequence rather than retroactively merged into Tests 10–16 here.

## Overall progression so far
The preserved sequence currently shows an early progression from:

`learn structure → predict data`

through:

`learn structure → exact residual → account for model cost`

then:

`learn contextual uncertainty → entropy-code directly → exact causal reconstruction`

then:

`learn whole-region structure → compact latent representation → exact correction information`

then:

`shared learned structure + decoder-known history → prediction → exact correction`

then:

`optimize structural context/capacity by total representation cost`

then:

`learned causal probability model → arithmetic-coded exact continuation`

with the Test 6 side branch asking:

`if latent structure is useful to store → can the latent/generative structure itself be discovered?`

then the scaling line adds:

`if structural models have fixed cost → does reuse over scale amortize that cost?`

then Test 8 strengthens that question with:

`real corpus + independent train/held-out split + decoder-valid hybrid selection + exact causal reconstruction`

Test 9 adds:

`GPU-scaled training/inference while keeping causal decoding sequential`

Test 10 reconnects that mature experimental line to direct probability coding:

`full learned uncertainty → deterministic range coding → exact causal continuation`

Test 12 introduces:

`shared initialization → online learning from stream history → potentially self-reconstructed model state`

Test 13 adds:

`latent representation → direct downstream computation without reconstructing the original representation`

and Test 14 extends that into:

`model weights → latent-state container → reconstructable computational tensors`,

while exposing that shape-compatible executability is not enough: reconstruction fidelity must be measured explicitly.

The remaining codec problem is to turn these ideas into complete representations and synchronized encoder/decoder protocols over equivalent coding domains. Subsequent historical reviews will continue reconstructing how that issue and the broader architecture evolved.