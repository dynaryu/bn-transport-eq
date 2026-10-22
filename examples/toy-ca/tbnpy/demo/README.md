# Toy California tbnpy model

`model.json` contains resolved variables, parent/child column orders, instance
parameters, network assumptions and source provenance. `model.py` constructs
tbnpy variables and probability objects from this snapshot, using the PyTorch
probability class each function's `implementation` names (`src/builder/kernels/`).

From the repository root, with the quakenet environment active:

```sh
python -m src build toy-ca/demo
python examples/toy-ca/tbnpy/demo/model.py --samples 2000 --device cpu
```

The second command writes `sample_summary.json`. To use the model in Python,
import `define_variables()` and `define_probs(varis, device="cpu")` from
`model.py`, then pass the probability dictionary to `tbnpy.inference.sample`.
Inference queries use keys in the probability dictionary; individual variables
are listed in each object's `childs`. Forward results' `Cs` columns follow that
child order. All samples and likelihood evaluations use PyTorch tensors, with
ndtools evaluating graph performance on the CPU.

There are 46 variables and five probability objects:

| Probability object | Child variables |
|---|---|
| Ground motion | All ten bridge SA(1.0) variables together; a root, with the Event row (Mw, epicentre, depth) as fixed data |
| Fragility | Ten five-state damage variables |
| Usability | Ten binary bridge variables |
| Edge open | Ten binary edge variables |
| System state | Six binary events: the nearest key destination reached within 0.5 hours of the intact time |

Toy-ca currently specifies one IM and independent total GMM residuals; grouping
the field in one probability object preserves that model without asserting a
new spatial or cross-IM correlation. Additional IMs require their coefficients
and joint residual model. The demo reuses eastern North America coefficients
in California for communication only. No hazard or fragility discretisation is
introduced.

`rsr/<origin>/target_add_h=<h>/` holds survival/failure reference tensors, readable
rules, extraction metrics and metadata for each origin at one threshold. RSR searches with a
50/50 independent proposal over edge states, solely to find topology rules.
The BN supplies the actual edge samples. Unclassified samples are evaluated by
ndtools. The system value rsr sees is the binary state itself; the travel time
ndtools computes is internal to that judgement.

Regenerate after changing source data. A fingerprint associates rules with their
network, edge order, destinations and analysis; the threshold is stored beside
them. Because the state is monotone in the threshold, a query at another
`target_add_h` reuses the survival rules of the nearest stricter threshold and the
failure rules of the nearest looser one, and ndtools decides the rest.
`python -m src rules toy-ca/demo --target-add-h 0.25` adds a threshold's own rules.
`--no-rules` on `build` skips extraction and uses ndtools directly. Queries, the
page and the chat build the same model in memory and sample it through tbnpy.
