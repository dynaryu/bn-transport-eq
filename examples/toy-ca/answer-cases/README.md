# Growing the answer cases

Copy one JSON file and rename it to match a unique `id`. Keep one question family
per file; add paraphrases to `questions`. Context must identify the scenario and
failure requirement. `context.defaults` holds the scenario settings the user sees.
Do not place expected conclusions or tool arguments in context.

`expected.outcome` is `compute`, `clarify` or `unsupported`.
`expected.tool_arguments` is the expected sequence of `run_query` arguments:
an ordinary query, `analysis` or `clarification`. Computation plans
use `analysis.kind` = `compare`, `importance`, `paths` or `voi`.
Probability questions use event queries. Before-and-after scenarios use separate
queries and report both probabilities. Expectations and explicitly requested
probability differences are valid if their meaning and units are stated; a
difference is not itself an event probability. Avoid arbitrary composite scores.
Clarifications name a question and stable `missing` labels. `must_not` records
semantic errors for colleague review, not just string matching.

For ambiguous terms such as "most important", provide a clarification with
`choices` (each with a `label` and mathematical `definition`) and wait for the
user. A follow-up case can put the earlier exchange in `context.history` and
test the selected query. VOI remains valid as an expected loss reduction.

Choose `split: few_shot` for a reviewed demonstration, or `held_out` for evaluation.
Only few-shot cases enter prompts. Never move held-out cases into prompts merely
to improve measured agreement. Sampling controls are ignored when comparing plans;
causal conditions, destinations, thresholds and comparison direction are retained.
Canonical agreement is conservative: equivalent plans may still need human review.

Suggested additions: topology-only versus hazard-driven importance; observed
versus forced bridge states; N1 versus either destination; AND versus OR; costly
versus free inspections; imperfect inspections; no beneficial inspection;
unknown origins; missing hazard inputs; and a second network. Synthetic losses
must be labelled as assumptions, not engineering estimates.

Commands:

```sh
python -m src evaluate
python -m src evaluate --live
python -m pytest src/inference/tests src/llm/tests src/app/tests
```

The first command checks case contracts without an LLM. The second uses the
configured provider and logs held-out translations under `records/queries.jsonl`;
it does not sample. Fixtures do not pin probability values:
use versioned model inputs and controlled seeds for numerical regression tests.
