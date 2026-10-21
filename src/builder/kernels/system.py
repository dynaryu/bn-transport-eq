"""System kernels: link states -> each system event's performance, and whether it survives.

A system event is a row of the network's SystemEvent table: an origin, a
destination, and what counts as surviving. Both kernels judge each sample
against the intact network (every link open), computed once per call, and read
the whole Edge plate per sample, so every event in one draw sees the same
network. Graphs are built dense (V x V), which is right for networks of a few
hundred nodes and the reason these are not meant for city-scale graphs.
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
        self.intact = np.ones(len(rows), bool)
        self.od = [(self.index[r["origin"]], self.index[r["destination"]]) for r in call.rows]
        for (o, d), r in zip(self.od, call.rows):
            if o == d:
                raise ValueError(f"system event {r['id']} has the same origin and destination")

    def arcs(self, live: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc tails, heads and weights of the links in ``live``, both ways for an undirected one."""
        back = live & ~self.directed
        return (np.concatenate([self.u[live], self.v[back]]),
                np.concatenate([self.v[live], self.u[back]]),
                np.concatenate([weight[live], weight[back]]))


class ShortestPath:
    @staticmethod
    def _times(g: _Topology, live: np.ndarray, minutes: np.ndarray, origins: list[int]) -> list[float]:
        tails, heads, w = g.arcs(live, minutes)
        dense = np.full((g.size, g.size), np.inf)
        np.minimum.at(dense, (tails, heads), w)                # parallel links: the faster one
        dist = dijkstra(csgraph_from_dense(dense, null_value=np.inf), directed=True, indices=origins)
        where = {o: i for i, o in enumerate(origins)}
        return [dist[where[o], d] for o, d in g.od]

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        g = _Topology(call)
        minutes = np.asarray([r["length_km"] for r in g.rows], float) / call.params["speed_kph"] * 60.0
        origins = sorted({o for o, _ in g.od})
        intact = np.asarray(self._times(g, g.intact, minutes, origins))
        never = [r["id"] for r, t in zip(call.rows, intact) if not np.isfinite(t)]
        if never:
            call.warnings.append(f"{len(never)} shortest-path event(s) have no route even with every link open, "
                                 f"so they always fail: {', '.join(never[:5])}")
        limit = intact * (1.0 + call.column("travel_time_tolerance")[0])
        out = np.full((call.n, len(call.rows)), np.inf)
        for s in range(call.n):
            out[s] = self._times(g, g.open[s], minutes, origins)
        survived = np.isfinite(out) & (out <= limit[None, :] * (1 + 1e-12))
        return {"travel_time": out, "state": survived.astype(int)}


class MaxFlow:
    @staticmethod
    def _flows(g: _Topology, live: np.ndarray, capacity: np.ndarray) -> list[float]:
        tails, heads, c = g.arcs(live, capacity)
        keep = tails != heads
        graph = csr_matrix((c[keep], (tails[keep], heads[keep])), shape=(g.size, g.size), dtype=np.int32)
        graph.sum_duplicates()                                 # parallel links: capacities add
        return [float(maximum_flow(graph, o, d).flow_value) for o, d in g.od]

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        g = _Topology(call)
        capacity = np.asarray([r["capacity"] for r in g.rows], np.int32)
        threshold = call.column("min_flow")[0]
        intact = np.asarray(self._flows(g, g.intact, capacity))
        never = [r["id"] for r, f, t in zip(call.rows, intact, threshold) if f < t]
        if never:
            call.warnings.append(f"{len(never)} max-flow event(s) cannot reach min_flow even with every link open, "
                                 f"so they always fail: {', '.join(never[:5])}")
        out = np.zeros((call.n, len(call.rows)))
        cache: dict[bytes, list[float]] = {}
        for s in range(call.n):
            key = g.open[s].tobytes()
            if key not in cache:                               # identical link states, identical flows
                cache[key] = self._flows(g, g.open[s], capacity)
            out[s] = cache[key]
        return {"max_flow": out, "state": (out >= threshold[None, :]).astype(int)}
