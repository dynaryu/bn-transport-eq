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

There are 52 variables and six probability objects:

| Probability object | Child variables |
|---|---|
| Ground motion | All ten bridge SA(1.0) variables together; a root, with the Event row (Mw, epicentre, depth) as fixed data |
| Fragility | Ten five-state damage variables |
| Usability | Ten binary bridge variables |
| Edge open | Ten binary edge variables |
| Travel time | Six origin travel times, in hours; infinity when disconnected |
| System state | Six binary events: travel within 0.5 hours of the intact time |

Toy-ca currently specifies one IM and independent total GMM residuals; grouping
the field in one probability object preserves that model without asserting a
new spatial or cross-IM correlation. Additional IMs require their coefficients
and joint residual model. The demo reuses eastern North America coefficients
in California for communication only. No hazard or fragility discretisation is
introduced.

`rsr/<origin>/` holds survival/failure reference tensors, readable rules,
extraction metrics and model metadata for each origin. RSR searches with a
50/50 independent proposal over edge states, solely to find topology rules.
The BN supplies the actual edge samples. Unclassified samples are evaluated by
ndtools. Extraction diagnostics use the binary state as the system value
because RSR 0.2.0 cannot round infinite travel times. Travel-time outputs retain
infinity for disconnected configurations.

Regenerate after changing source data. A snapshot fingerprint associates rules
with their network, edge order, destinations and thresholds. `--no-rules` skips
extraction and uses ndtools directly. The existing app/query engine still uses
its original sampler; this generated model is sampled directly through tbnpy.
