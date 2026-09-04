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
Original script: `artifacts/test2.py`

Confirmed by project history to follow Tests C and D. Review pending. Initial inspection shows a return to whole-file fixed-size blocks using a neural encoder/decoder bottleneck and an exact modulo-256 residual, with intended storage accounting of model + residual + metadata.

## Historical context
The earliest work was not originally expected to become a long experiment series, so initial experiments were lettered rather than organized as a mature numbered research program. Tests C and D precede numbered Test 2. The first roughly fourteen tests were developed in a very compressed early research period, making this retrospective documentation particularly useful for reconstructing the actual conceptual progression.

## Overall progression so far
The preserved sequence currently shows an early progression from:

`learn structure → predict data`

through:

`learn structure → exact residual → account for model cost`

and then:

`learn contextual uncertainty → entropy-code directly → exact causal reconstruction`

Test 2 and subsequent reviews will extend this timeline as they are examined.