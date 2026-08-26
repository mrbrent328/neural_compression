# E36 Archaeology Record

## Why E36 matters

E36 is the bridge from the earlier latent/geometry work toward the current persistent-relational question.

The base E36 prototype asks whether a moving observer can build useful persistent knowledge from only locally visible information. Hidden geometry is not passed to memory. The memory can update existing chunks, create new chunks, expand when its current compact representation no longer explains an observation, and merge compatible chunks. Its explicit cost model distinguishes persistent knowledge from raw observation history.

The topology version is especially important because repeated observation of a static region is treated as confirmation rather than additional stored history. A chunk is represented by compact sufficient statistics rather than a retained observation history. Expansion occurs only when current knowledge cannot explain the new observation; compatible chunks can later merge.

## E36 v4b — relational transition

The relational version changes the conceptual primitive from local chunks to persistent landmarks plus relations.

A landmark is identified from intrinsic perceptual state rather than absolute position. Relations record how observations connect through observer movement. Absolute location is not stored.

The experiment performs two traversals while retaining memory between them. This makes the second traversal a reuse test rather than simply another accumulation of observations.

The memory boundary explicitly excludes absolute world coordinates, world IDs, hidden geometry, and position-derived features. It receives local relative observations, intrinsic features, distance, observer motion, and relative rotation change.

The source describes the intended progression as persistent local landmarks connected by relational transition information. The second-pass reuse fraction and relation growth are measured explicitly.

## Interpretation

The most useful abstraction is:

`local observation → persistent units → reuse across time → relations between units → changed observation`

E36 does not establish that its hand-built memory mechanism is optimal. Its value is that it turns the persistent-structure idea into an inspectable mechanism and supplies the conceptual ingredients that E37 formalizes.

## Evidence source

Project artifacts:

`E36_local_observer.py`

`E36_local_topology_v3.py`

`E36_local_relational_v4b.py`

The complete source files are still retained in the ChatGPT Project / user's local forensic archive. Only partial source chunks were exposed during this pass, so they are intentionally not fabricated into Git from incomplete snippets.

## Relation to E37

E36 v4b supplies the conceptual bridge to E37 MPRS: intrinsic local signatures, persistent units, sparse relations, observer motion as transition information, and reuse across traversal.

E37 then changes the question from “can this kind of persistent structure work?” to “what is the smallest tested persistent relational structure that remains sufficient when the observation changes?”
