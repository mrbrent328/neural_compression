# E39 — Persistent Relational Sufficiency

## Question

How much information can be removed from an observation stream while a persistent relational representation retains the ability to answer previously unseen questions about the same underlying system?

E39 follows the product-direction insight that relational memory may be useful not merely as a retrieval optimization, but as a compact persistent representation. It tests the representation itself before adding an LLM layer.

## Design

The experiment constructs a deterministic observation corpus, converts it into a persistent relational graph, then evaluates questions that were held out from construction. The evaluation condition has access only to the graph, never the original observations. A source-grounded oracle provides the reference answer.

The graph is evaluated at multiple relation budgets. This measures the tradeoff between retained relational structure and retained answering capability.

## Controls

The full graph is the primary representation condition. Reduced graphs are produced deterministically from the same full graph. Questions are generated independently of the graph-construction process and are not used to select relations.

The experiment reports both exact answer accuracy and whether the graph can provide a supporting relation chain for each answer.

## Boundary

This is not an LLM experiment and does not test semantic understanding. It tests whether the persistent relational representation itself preserves enough structure to answer new questions about the observed system.

## Run

```bash
python run_e39.py
```

The run writes a JSON result artifact containing the corpus, graph statistics, question results, relation-budget sweep, and exactness checks.
