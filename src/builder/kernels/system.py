"""System kernels: link states -> origin-destination performance.

Both read the whole Edge plate per sample, so every OD pair in one draw sees the
same network. Graphs are built dense (V x V), which is right for networks of a
few hundred nodes and the reason these are not meant for city-scale graphs.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense, dijkstra, maximum_flow

from . import Call


class _Topology:
    def __init__(self, call: Call) -> None:
        nodes = [row["id"] for row in call.tables["Node"]]
        self.index = {node: i for i, node in enumerate(nodes)}
        self.size = len(nodes)
        edges = {row["id"]: row for row in call.tables["Edge"]}
        order = call.parents["open"][0]                       # via *: the same edge list for every row
        rows = [edges[e] for e in order]
        self.u = np.asarray([self.index[r["source"]] for r in rows], int)
        self.v = np.asarray([self.index[r["target"]] for r in rows], int)
        self.directed = np.asarray([r["directed"] for r in rows], bool)
        self.rows = rows
        self.open = call.inputs["open"][0].astype(bool)       # (n, E)
        self.od = [(self.index[r["origin"]], self.index[r["destination"]]) for r in call.rows]
        for (o, d), r in zip(self.od, call.rows):
            if o == d:
                raise ValueError(f"OD pair {r['id']} has the same origin and destination")

    def arcs(self, sample: int, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc tails, heads and weights of the open links, both ways for an undirected one."""
        live = self.open[sample]
        back = live & ~self.directed
        return (np.concatenate([self.u[live], self.v[back]]),
                np.concatenate([self.v[live], self.u[back]]),
                np.concatenate([weight[live], weight[back]]))


class ShortestPath:
    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        g = _Topology(call)
        minutes = np.asarray([r["length_km"] for r in g.rows], float) / call.params["speed_kph"] * 60.0
        origins = sorted({o for o, _ in g.od})
        where = {o: i for i, o in enumerate(origins)}
        out = np.full((call.n, len(call.rows)), np.inf)
        for s in range(call.n):
            tails, heads, w = g.arcs(s, minutes)
            dense = np.full((g.size, g.size), np.inf)
            np.minimum.at(dense, (tails, heads), w)            # parallel links: the faster one
            graph = csgraph_from_dense(dense, null_value=np.inf)
            dist = dijkstra(graph, directed=True, indices=origins)
            out[s] = [dist[where[o], d] for o, d in g.od]
        return {"travel_time": out}


class MaxFlow:
    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        g = _Topology(call)
        capacity = np.asarray([r["capacity"] for r in g.rows], np.int32)
        out = np.zeros((call.n, len(call.rows)))
        cache: dict[bytes, list[float]] = {}
        for s in range(call.n):
            key = g.open[s].tobytes()
            if key not in cache:                               # identical link states, identical flows
                tails, heads, c = g.arcs(s, capacity)
                keep = tails != heads
                graph = csr_matrix((c[keep], (tails[keep], heads[keep])), shape=(g.size, g.size), dtype=np.int32)
                graph.sum_duplicates()                         # parallel links: capacities add
                cache[key] = [float(maximum_flow(graph, o, d).flow_value) for o, d in g.od]
            out[s] = cache[key]
        return {"max_flow": out}
