"""Surgical interventions at the probability-object boundary, before descendants draw."""
from src.builder.kernels import Probability


class Constant(Probability):
    """do(child = value): the childs are fixed, whatever their parents were (deterministic, a root)."""

    def __init__(self, childs, values, device="cpu"):
        """
        childs : the Variables the intervention fixes
        values : the value of each, in the same order (a state index for a discrete variable)
        """
        super().__init__(childs, [], device)
        assert len(values) == len(childs), "one value per child"
        self.values = list(values)

    def form(self):
        return "do(" + ", ".join(f"{v.name} = {c:g}" for v, c in zip(self.childs, self.values)) + ")"

    def evaluate(self, Cs_pars):
        """
        Cs_pars : (N, 0), a root
        Returns (N, n_childs), every row the fixed values
        """
        return self.tensor(self.values)[None, :].expand(len(Cs_pars), -1)


def apply(probs, assignments):
    for key, prob in list(probs.items()):
        selected = [v.name for v in prob.childs if v.name in assignments]
        if not selected:
            continue
        if len(prob.childs) != 1:
            raise ValueError("Node interventions on a joint probability object are not supported; "
                             "its other children must retain their joint law")
        constant = Constant(prob.childs, [assignments[v.name] for v in prob.childs], device=str(prob.device))
        constant.spec_metadata = {**prob.spec_metadata, "role": "intervention", "constant": constant.values}
        probs[key] = constant
