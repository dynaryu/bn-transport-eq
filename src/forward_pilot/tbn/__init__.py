"""The pilot's Bayesian network, expressed in tbnpy.

One module per node class, following the convention in tbnpy's own examples.
``model.build`` assembles them into the ``(varis, probs)`` pair tbnpy's
inference functions expect.
"""

from .model import build, query_nodes_for  # noqa: F401
