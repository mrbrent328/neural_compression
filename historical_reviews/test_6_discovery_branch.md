# Historical Review — Test 6 / Latent Structure Discovery Branch

## Sources
- `artifacts/test6.py`
- `artifacts/discover_law2.py`
- `artifacts/discover_law3.py`
- `artifacts/blind_law_suite.py`

## Historical origin
This branch began directly from the Neural Compression latent-representation question.

The motivating thought was:

> If a system can store latent structure, can it find the latent structure itself?

That turns the compression problem around. Instead of only asking how a learned state can cheaply represent observations, the branch asks whether the system can infer the compact relationship or generating process responsible for those observations.

Conceptually:

`observations → latent structure → compact representation`

becomes

`observations → discover latent structure/generating rule → store or use the rule`

This is therefore a side branch, but not an unrelated one. It tests whether the latent information Neural Compression hopes to exploit can be explicitly recovered rather than remaining implicit inside model weights.

## Test 6 — first law-discovery experiment
`test6.py` is titled `SCIENTIFIC LAW DISCOVERY — EXPERIMENT 1`.

It generates noisy observations of time and position from a hidden free-fall-like law:

`position = 4.905 * time^2`

The discovery routine is not given that equation. It receives observations, a library of candidate mathematical basis functions, and searches one- and two-term linear combinations. The basis library includes constants, powers, inverse, square root, sine, cosine, logarithm, and exponential terms.

The experiment uses 250 observations, a 70/30 train/test split, measurement noise, and an additional 500 fresh random points for post-discovery generalization testing.

### What it establishes
The important conceptual step is not merely symbolic regression performance. It is the attempt to replace an implicit learned statistical representation with an explicit compact generating relationship.

This reframes the compression intuition as:

**perhaps the shortest useful description of observations is the rule that generates them.**

### Limitation
The correct `x^2` functional form is already present in the candidate library. The system discovers which available term and coefficient explain the observations; it does not invent unrestricted mathematics. Candidate ranking is primarily by training error, with complexity only secondary, so the first experiment is not yet a strong minimum-description-length search.

## `discover_law2.py` — blind multivariable discovery
The next script makes the discovery problem substantially harder.

The hidden relationship is:

`output = 2.75*x^2 - 1.50*y + 0.80*x*z + 3.20`

The system is not told which variables matter, how many terms matter, or which mathematical operations matter. Variable `w` is deliberately irrelevant.

The candidate vocabulary expands to powers, roots, logarithms, trigonometric functions, exponentials, inverse terms, pairwise products, ratios, and nonlinear interactions. Search allows up to five terms and uses a complexity penalty. It also uses noisy observations, held-out evaluation, numerical stabilization, and fresh-data testing.

### Progression
This moves from:

`find one known-shape law among simple alternatives`

toward:

`identify relevant variables + operations + interactions + coefficients while rejecting irrelevant structure`.

That is much closer to discovering which latent relationships actually matter.

## `discover_law3.py` — complexity-aware structure selection
`discover_law3.py` explicitly describes itself as complexity-aware symbolic regression.

It retains the blind multivariable problem but assigns different complexity costs to mathematical features and evaluates multiple complexity penalties (`lambda`). Equation complexity includes feature complexity, coefficient-magnitude treatment, and explicit term-count cost.

This is especially relevant to Neural Compression because the question becomes not simply which equation predicts best, but which explanatory structure provides the best tradeoff between fit and descriptive complexity.

Conceptually:

`score ≈ prediction error + λ * representation complexity`

This is closely aligned with the project's recurring model-cost/residual-cost tradeoff, now applied to explicit mathematical laws rather than neural predictor storage.

### Methodological note
The candidate mathematical vocabulary is still supplied by the experiment. The system searches combinations within that vocabulary rather than constructing arbitrary symbolic programs from first principles.

## `blind_law_suite.py` — generalization across law families
The branch then broadens from one hidden relationship to a suite of hidden generating processes.

The preserved suite includes datasets named:

- `QUADRATIC`
- `MIXED`
- `INVERSE`
- `EXPONENTIAL`
- `TRIG`

The discovery engine receives observations rather than the generating equations, searches a broad mathematical feature vocabulary, ranks candidates with a complexity-aware score, uses held-out observations, and evaluates discovered models on fresh data.

This changes the question from whether one chosen law can be recovered to whether the same discovery procedure can recover compact structure across qualitatively different generating families.

A methodological boundary remains: recovery is only possible when the necessary functional structure is expressible in the supplied feature vocabulary. For example, a generating function whose exact form is absent from the library may only be approximated by available terms.

## Connection back to Neural Compression
The branch exposes a deeper version of the original compression hypothesis.

Earlier experiments ask whether learned models can exploit regularity so fewer bits need to describe data. The discovery branch asks whether the regularity itself can become the representation.

There are three increasingly strong possibilities:

1. **Predictive structure** — a model predicts likely data and entropy/residual coding stores what it misses.
2. **Latent representation** — a compact learned state represents important structure and correction information restores exact detail.
3. **Explicit generative structure** — infer a compact rule/process that explains or generates the observations, then represent the rule plus whatever exceptions/noise remain.

The third possibility is the conceptual contribution of this side branch.

This anticipates later Neural Compression themes involving structural representation, state geometry, relational organization, and selecting representations according to their descriptive efficiency.

## Retrospective conclusion
**Test 6 begins the latent-structure discovery branch: if latent structure can be stored, perhaps it can first be discovered and represented directly as the generating relationship behind the data.**

The follow-on scripts progressively strengthen that question through irrelevant-variable rejection, interaction discovery, explicit complexity penalties, and multiple hidden law families.

This branch does not establish a universal scientific-law discovery engine. It is symbolic regression over supplied mathematical vocabularies. Its historical importance is conceptual: it converts the compression question from "how cheaply can we encode observations?" toward "can we discover the compact process that makes those observations necessary?"

### Historical transition

`learn/store latent structure`

→

`discover latent structure`

→

`prefer compact explanatory/generative structure`

→

`test whether the same discovery principle generalizes across structural families`.