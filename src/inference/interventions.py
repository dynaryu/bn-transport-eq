"""Surgical interventions at the probability-object boundary, before descendants draw."""
from src.builder.kernels import Probability


class Constant(Probability):
    def form(self):
        values = ", ".join(f"{v.name} = {c:g}" for v, c in zip(self.childs, self.spec["constant"]))
        return f"do({values})"

    def evaluate(self, pars):
        return self.tensor(self.spec["constant"])[None, :].expand(len(pars), -1)


def apply(probs, assignments):
    for key, prob in list(probs.items()):
        selected = [v.name for v in prob.childs if v.name in assignments]
        if not selected:
            continue
        if len(prob.childs) != 1:
            raise ValueError("Node interventions on a joint probability object are not supported; "
                             "its other children must retain their joint law")
        spec = {**prob.spec, "role": "intervention", "constant": [assignments[v.name] for v in prob.childs]}
        probs[key] = Constant(prob.childs, [], spec, device=str(prob.device))
