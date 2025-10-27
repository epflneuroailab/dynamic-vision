import random
from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
import networkx as nx


def get_probabilistic_hierarchy(edge_list, num_runs=100):
    from simanneal import Annealer

    class HierarchyAnnealer(Annealer):
        def __init__(self, state, edges):
            self.edges = edges  # List of (u, v) tuples
            super().__init__(state)

        def move(self):
            # Randomly swap two elements in the node order
            i, j = random.sample(range(len(self.state)), 2)
            self.state[i], self.state[j] = self.state[j], self.state[i]

        def energy(self):
            # Cost = number of violated edges (u should come before v)
            position = {node: i for i, node in enumerate(self.state)}
            violations = sum(1 for u, v in self.edges if position[u] > position[v])
            return violations

    edge_list = list(set(edge_list))
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    nodes = list(G.nodes())
    edges = list(G.edges())

    # Dictionary to track how often each node appears at each level
    level_counts = defaultdict(lambda: defaultdict(int))

    def _sim():
        initial_state = random.sample(nodes, len(nodes))
        sa = HierarchyAnnealer(initial_state, edges)
        sa.set_schedule(sa.auto(minutes=0.1))
        best_state, cost = sa.anneal()
        return best_state

    results = Parallel(n_jobs=1)(delayed(_sim)() for _ in range(num_runs))

    for state in results:
        for i, node in enumerate(state):
            level_counts[node][i] += 1

    max_level = len(nodes)
    heatmap = np.zeros((len(nodes), max_level))
    node_index = {node: i for i, node in enumerate(nodes)}

    for node, levels in level_counts.items():
        total = sum(levels.values())
        for level, count in levels.items():
            heatmap[node_index[node], level] = count / total

    aver_hier = (heatmap * np.arange(1, max_level + 1)).sum(axis=1) / max_level

    return nodes, heatmap, aver_hier