import unittest
from sigmaepsilon.math.topology import Graph
import networkx as nx


class TestGraph(unittest.TestCase):
    def test_Graph(self):
        grid = nx.grid_2d_graph(5, 5)  # 5x5 grid
        G = Graph(grid)
        G.adjacency_matrix()
        G.pseudo_peripheral_nodes()
        G.rooted_level_structure()

        seed = 13648  # Seed random number generators for reproducibility
        G = Graph(nx.random_k_out_graph(10, 3, 0.5, seed=seed))
        G.pseudo_peripheral_nodes()
        G.rooted_level_structure()


if __name__ == "__main__":
    unittest.main()
