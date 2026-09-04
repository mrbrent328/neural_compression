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

## Historical context
The earliest work was not originally expected to become a long experiment series, so initial experiments were lettered rather than organized as a mature numbered research program. Tests C and D precede numbered Test 2. The first roughly fourteen tests were developed in a very compressed early research period, making this retrospective documentation particularly useful for reconstructing the actual conceptual progression.

Test 6 temporarily branches away from direct codec development, but the branch follows directly from the compression hypothesis: perhaps the cheapest representation of observations is the compact relationship or process that generated them.

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

The remaining codec accounting problem is to compare complete representations over genuinely equivalent coding domains, especially eliminating or charging source-specific shared prefix/training information. Subsequent historical reviews will continue reconstructing how that issue and the broader architecture evolved.