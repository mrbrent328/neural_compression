# Historical Review — Test 5

## Source
- Original script: `artifacts/test5.py`
- Historical position: follows Test 4.

## Purpose
Test 5 turns the learned causal next-byte model into a real arithmetic-coded continuation stream and verifies that the decoder can reproduce the held-out portion symbol by symbol using the same probability path as the encoder.

It revisits the probability-coding direction introduced in Test D, but now after Tests 3–4 clarified decoder-reconstructible context and the cost of learned model capacity.

## Design
- Fixed context width: 8 bytes.
- Learned embedding: 16 dimensions per byte.
- Hidden layers: 64 → 64.
- Output: 256 categorical logits.
- Training: first 75% of the file, using up to 500,000 sampled positions.
- Held-out coding region: final 25%.
- Entropy coder: custom 32-bit arithmetic encoder/decoder.
- Neural probabilities are deterministically converted to integer frequencies before coding.

The encoder advances its context using the true encoded symbol. The decoder generates the same neural probability table, decodes one symbol, appends it, and then advances its context using the decoded symbol.

## Major progression from Test 4
Tests 2–4 verified exact residual arithmetic but did not run a complete independent sequential decoding path for the learned continuation.

Test 5 does.

The held-out stream is reconstructed causally as:

`shared prefix → neural probabilities → arithmetic-decode symbol → append decoded symbol → update context`

This substantially strengthens the codec mechanism. The decoder no longer depends on precomputed predictions taken from the original held-out data.

## Probability-coding significance
Test 5 removes the residual+zlib stage for the held-out region and lets the neural probability distribution directly determine code length through arithmetic coding.

The script reports:
- average neural cross-entropy in bits/byte;
- theoretical payload size implied by that cross-entropy;
- actual arithmetic payload size;
- actual coded rate.

This provides a useful sanity check between model uncertainty and realized entropy-coded storage.

## Exactness
The decoder reconstructs the full byte sequence by combining the supplied prefix with the decoded continuation. The script compares:
- exact byte equality;
- original and reconstructed lengths;
- SHA-256 of the complete original and reconstruction.

If reconstruction fails it reports the first differing position and local context.

This is the strongest end-to-end causal decoding verification in the numbered sequence so far.

## Remaining accounting limitation
The whole-file accounting problem remains.

The learned representation counts only:

`arithmetic payload for final 25% + compressed model + 1,024-byte metadata allowance`

while the first 75% is supplied directly to the decoder as an uncharged prefix.

The zlib baseline, however, is zlib of the full file. Therefore claims comparing the learned total with full-file zlib are not equivalent-domain whole-file compression comparisons.

The script's wording that the neural payload 'beats zlib on the held-out data' is also not literally supported by the baseline variable used there, because `raw_zlib` contains the full-file zlib stream rather than zlib of only the held-out bytes.

## Packaging limitation
Model storage is measured by concatenating raw tensor bytes and zlib-compressing them. Metadata is still represented by a fixed 1,024-byte allowance rather than a concrete independently loadable package format.

Thus the causal arithmetic stream and decoder are real, but the complete source-specific model/prefix/package representation is not yet a standalone whole-file codec artifact.

## Preserved-result status
The script is preserved. No authoritative historical numerical output is used in this review, so no claim is made about whether the original Test 5 run beat zlib numerically.

## Retrospective conclusion
**First proper causal neural arithmetic codec in the numbered sequence; exact decoder path validated; shared-prefix whole-file accounting still unresolved.**

Historical transition:

`optimize learned context/model cost`

→

`use the learned probability distribution directly as a lossless arithmetic-coded bitstream`

Test 5 therefore closes an important mechanism gap even though it does not yet close the complete-representation accounting gap.