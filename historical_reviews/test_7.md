# Historical Review — Test 7

## Source
- `artifacts/test7.py`

## Purpose / question
Test 7 asks whether predictive-compression model overhead becomes increasingly economical as dataset size grows.

Its core hypothesis is:

> A structural model may be expensive at small scale but become worthwhile if its fixed cost is amortized across enough data.

The script evaluates datasets from 100 KB through 5 MB across structured text, source-code-like data, and mathematical text.

Methods compared:
- zlib baseline;
- previous-byte prediction + exact residual;
- context model + exact residual;
- neural predictor + exact residual;
- a hybrid previous-byte/neural predictor.

The script explicitly reports residual size, model cost, metadata cost, and total representation size.

## Experimental design
The largest synthetic dataset is generated once. Context and neural models are trained once on the first 75% of that 5 MB dataset, and then reused across all smaller curve points.

The intended reason is good: hold predictor/model cost fixed so that the effect of data volume on amortization can be isolated rather than retraining a different model at every size.

Neural configuration includes:
- context size 4;
- hidden size 64;
- up to 300,000 training samples;
- 500 epochs;
- raw parameter-cost estimate;
- 1,024 bytes metadata.

For each curve point, the final 25% is predicted and encoded as an exact modulo-256 residual compressed with zlib.

The script then searches for empirical break-even points where a predictive representation becomes smaller than zlib.

## Preserved evidence
No authoritative `test4_scaling_results.json` or `.csv` output is preserved in the current repository. The surviving evidence is therefore the code and experimental design, not historical numerical outcomes.

## Important methodological problems

### 1. Smaller held-out regions leak into training
The fixed context/neural models are trained on the first 75% of the largest 5 MB dataset: 3.75 MB.

Every smaller curve point is a prefix of that same full dataset. Therefore, for sizes below 5 MB, the local final 25% that the script labels as held-out is generally still inside the 3.75 MB region already used to train the fixed model.

Examples:
- 100 KB point: local held-out region = 75–100 KB, which was in model training;
- 1 MB point: local held-out = 750 KB–1 MB, also in model training;
- 2 MB point: local held-out = 1.5–2 MB, also in model training.

Only the final quarter of the full 5 MB point lies genuinely outside the fixed training region.

This means the scaling curve does not cleanly measure generalization-versus-size.

### 2. Unequal coding domains versus zlib
The predictive totals include only:

`compressed residual of final 25% + model + metadata`

but the zlib baseline is:

`zlib(full 100% dataset)`

Therefore the printed comparison and break-even analysis do not establish whole-file compression superiority.

A fair standalone comparison would need either:
- all source-specific prefix/training information charged to the learned representation; or
- both learned and baseline methods evaluated on exactly the same held-out domain under an explicitly shared/pretrained-model assumption.

### 3. Hybrid predictor uses target-byte information
The hybrid computes the neural and previous-byte errors against the actual held-out byte and then chooses whichever predictor happened to be closer.

That selection is not reproducible by a decoder without either:
- transmitting selector information; or
- replacing it with a deterministic selector based only on decoder-known state.

As implemented, the hybrid is an oracle combination and is not a valid standalone codec.

### 4. Neural model cost is only an estimate
Neural model storage is approximated as parameter count × 4 bytes rather than actual self-contained serialized/compressed model storage. This is useful as a rough fixed-cost estimate but is not yet a real packaged representation size.

## What Test 7 genuinely contributes
The central scientific question is important even though the implementation cannot prove the requested break-even curve.

It introduces the idea of **representation amortization**:

`total description cost = fixed structural model cost + variable unexplained/correction cost`

As data volume grows, a reusable structural representation can become economically attractive even if it is locally expensive.

That insight connects strongly to later Neural Compression work involving shared state, reusable structure, specialization, local strategy selection, and deciding where richer representations are worth their cost.

The experiment also shows a growing awareness that compression comparisons must include model and metadata overhead rather than only residual quality.

## Retrospective conclusion
**The Test 7 hypothesis survives; the empirical break-even curve does not.**

The experiment is historically valuable because it explicitly asks whether learned structural overhead can be amortized across increasing data volume. However, training leakage into smaller curve points, unequal coding domains versus zlib, and oracle hybrid selection prevent the preserved implementation from establishing a valid scaling/break-even result.

### Historical transition

`model overhead hurts`

→

`perhaps reusable structural overhead becomes economical over enough data`

→

`measure compression as representation economics rather than residual quality alone`

This is an early precursor to later shared-structure and adaptive-capacity ideas.