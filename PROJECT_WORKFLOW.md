# Neural Compression — Project Workflow

This is the standing workflow for Neural Compression research. It applies to every new research conversation unless explicitly changed by the user.

## One Experiment Per Conversation

Each research conversation is centered on exactly one experiment.

Do not bundle multiple experiments together. Follow-on experiments belong in later conversations and should be proposed from the evidence produced by the current one.

## Conversation Procedure

### 1. Review the persistent project state

At the beginning of a research conversation, review the current branch, the research log, the latest handoff file, and the relevant current experiment code/results before proposing the next step.

The repository is the durable research memory. Do not rely on conversational memory when the repository contains the authoritative record.

### 2. Propose one experiment

Use the accumulated evidence to propose one concrete next experiment.

State the scientific question clearly, explain why this experiment is the most useful next discriminator, and define what would count as informative evidence.

Do not start coding yet.

### 3. Discuss and agree on the experiment

Talk through the proposed experiment with the user. Resolve genuine ambiguities and revise the design when the discussion reveals a better test.

Once the user explicitly approves the experiment, stop re-planning it unless a concrete implementation problem requires a change.

### 4. Design the implementation

Describe how the approved experiment will be implemented, including inputs, outputs, controls, measurements, exactness requirements, and relevant failure checks.

Review the design with the user when useful.

### 5. Code the experiment

Once the design is approved, actually implement the experiment.

Do not merely describe code that should be written or imply that code has been written when it has not.

### 6. Commit to the canonical branch

Put the experiment in GitHub and commit it to the project's canonical branch.

The current canonical branch is `main`.

Give the user the exact repository path, branch, and commit after the commit is made.

### 7. User runs the experiment

The user runs the committed experiment on their local machine and returns the actual output.

Do not substitute simulated, assumed, or expected results for the user's real run.

### 8. Analyze the actual result

Review the output together.

Separate observations from interpretation. Preserve exactness checks and do not weaken tests merely to make an experiment pass.

Decide what the evidence supports, weakens, or leaves unresolved.

### 9. Leave a handoff for the next conversation

Before the conversation ends, create or update a concise handoff file containing the current research state, the result of the experiment, important implementation lessons, and the next unresolved decision.

The handoff should make it possible for a fresh conversation to resume without asking the user to reconstruct the prior discussion.

## Research Discipline

Keep observations, invariants, hypotheses, rejected explanations, open questions, current model, and next decision distinct.

Do not retroactively reinterpret old experiments as proving later hypotheses. Historical experiments should first be recorded according to what they actually tested, then connected to later ideas where the evidence supports that connection.

Prefer the smallest experiment that cleanly separates competing explanations.

Treat surprising negative results as information. Do not discard an experiment simply because it failed to produce the expected effect.

For exact-reconstruction experiments, exact reconstruction is a hard requirement. Never weaken or remove exactness checks merely to obtain a successful run.

## Branch Discipline

The research project uses one canonical working branch: `main`.

New experiments should be committed directly to `main` rather than creating another long-lived research branch.

Historical branch references may exist for archaeology or recovery, but they are not active experiment branches unless explicitly designated otherwise.

## Standing Principle

The purpose of the workflow is to maintain a continuous experimental chain:

`question → proposal → discussion → approved design → implementation → commit → real run → analysis → handoff → next question`

Each conversation should advance exactly one link in the research chain rather than repeatedly rebuilding context.
