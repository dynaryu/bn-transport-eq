"""One-step EVSI for an explicit finite loss table and binary inspection likelihoods."""
import itertools
import numpy as np

from . import engine


def from_distribution(states, probabilities, task):
    """Minimise losses before/after observation. Actions do not alter the prior state law.

    Action consequences must already be represented in the supplied loss table.
    This is a decision table, not an implicit simulation of repair or counterfactuals.
    """
    keys = ["|".join(map(str, s)) for s in states]
    for action in task["actions"]:
        if set(action["losses"]) != set(keys):
            raise engine.QueryError("Each action needs a loss for every state combination, keyed like '0|1'")
    losses = np.array([[a["losses"][k] for k in keys] for a in task["actions"]], float)
    if not np.isfinite(losses).all():
        raise engine.QueryError("Losses must be finite")
    p = np.asarray(probabilities, float)
    if p.shape != (len(states),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise engine.QueryError("State probabilities must be finite, nonnegative and sum to one")
    current = losses @ p
    best = int(current.argmin())
    rows = []
    for inspection in task["inspections"]:
        if inspection["node"] not in task["state_nodes"]:
            raise engine.QueryError("Inspection node must occur in state_nodes")
        accuracy = np.asarray(inspection["accuracy"], float)
        if accuracy.shape != (2, 2) or np.any(accuracy < 0) or not np.allclose(accuracy.sum(axis=1), 1):
            raise engine.QueryError("Inspection accuracy needs two probability rows, each summing to one")
        j = task["state_nodes"].index(inspection["node"])
        expected_loss, outcomes = 0., []
        for y in range(2):
            joint = p * accuracy[np.array(states)[:, j], y]
            py = float(joint.sum())
            if py == 0:
                continue
            posterior_losses = losses @ (joint / py)
            action = int(posterior_losses.argmin())
            expected_loss += py * float(posterior_losses[action])
            outcomes.append({"reported_state": y, "p": py, "action": task["actions"][action]["name"],
                             "expected_loss": float(posterior_losses[action])})
        evsi = max(0., float(current[best]) - expected_loss)
        rows.append({"node": inspection["node"], "evsi": evsi, "cost": inspection["cost"],
                     "net_value": evsi - inspection["cost"], "outcomes": outcomes})
    rows.sort(key=lambda r: r["net_value"], reverse=True)
    return {"kind": "voi", "loss_units": task["loss_units"], "current_action": task["actions"][best]["name"],
            "current_expected_loss": float(current[best]), "rows": rows,
            "recommendation": rows[0]["node"] if rows[0]["net_value"] > 0 else "do not inspect",
            "note": "One inspection at a time; action consequences come from the supplied loss table. "
                    "Monte Carlo point estimates, without a VOI confidence interval."}


def evaluate(task, run):
    names = task["state_nodes"]
    compiled = engine.compile_query(task["query"])
    nodes = {n["id"]: n for n in compiled.nodes()}
    for name in names:
        if name not in nodes or len(compiled.library.quantities[nodes[name]["quantity"]].get("states") or []) != 2:
            raise engine.QueryError("VOI state_nodes must be exact binary nodes, with two states (indices 0 and 1)")
    states = list(itertools.product((0, 1), repeat=len(names)))
    # Validate losses/likelihoods before sampling.
    from_distribution(states, np.full(len(states), 1 / len(states)), task)
    events = [{"name": "|".join(map(str, s)), "expression": {"all": [
        {"node": n, "op": "==", "value": v} for n, v in zip(names, s)]}} for s in states]
    query = {k: v for k, v in task["query"].items() if k not in ("events", "targets", "joint_events")}
    result = run({**query, "joint_events": events})
    answer = from_distribution(states, [e["p"] for e in result["events"]], task)
    answer["inference"] = result["inference"]
    return answer
