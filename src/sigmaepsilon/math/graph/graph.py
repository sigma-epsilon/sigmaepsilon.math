from ..linalg.sparse import csr_matrix
from .utils import rooted_level_structure, pseudo_peripheral_nodes

__all__ = ["Graph"]

try:
    import networkx as ntx
    import numpy as np

    try:
        adjacency_matrix = ntx.to_scipy_sparse_array
    except Exception:  # pragma: no cover
        adjacency_matrix = ntx.adjacency_matrix

    class Graph(ntx.Graph):
        """
        A subclass of `networkx.Graph`, extending its capabilities.
        See the documentation of `networkx` for the details on how to
        define graphs.

        Note
        ----
        If `networkx` is not installed, the class is `NoneType`, but the
        functionality it implements is still available, you just have to
        manage graph creation by yourself.

        Examples
        --------
        A basic example with `networkx`:

        >>> from sigmaepsilon.math.graph import Graph
        >>> import networkx as nx
        >>> grid = nx.grid_2d_graph(5, 5)  # 5x5 grid
        >>> G = Graph(grid)

        """

        def adjacency_matrix(self, *args, to_csr: bool = False, **kwargs) -> csr_matrix:
            """
            Returns the adjacency matrix of the graph.

            Parameters
            ----------
            to_csr : bool, Optional
                If `True`, the result of networkx.adjacency_matrix is
                returned as a csr_matrix.
            *args : Tuple, Optional
                Forwarded to networkx.adjacency_matrix
            **kwargs, dict, Optional
                Forwarded to networkx.adjacency_matrix

            Returns
            -------
            NumPy array, SciPy array or csr_matrix
                The adjacency representation of the graph.

            Examples
            --------
            >>> from sigmaepsilon.math.graph import Graph
            >>> G = Graph([(1, 1)])
            >>> A = G.adjacency_matrix()
            >>> print(A.todense())
            [[1]]

            """
            adj = adjacency_matrix(self, *args, **kwargs)
            return csr_matrix(adj) if to_csr else adj

        def rooted_level_structure(self, root: int = 0) -> dict[int, np.ndarray]:
            """
            Returns the rooted level structure (RLS) of the graph.

            The call is forwarded to `rooted_level_structure`, go there
            to read about the possible arguments.

            See Also
            --------
            :func:`rooted_level_structure`
            """
            return rooted_level_structure(csr_matrix(adjacency_matrix(self)), root)

        def pseudo_peripheral_nodes(self) -> np.ndarray:
            """
            Returns the indices of nodes that are possible candidates
            for being peripheral nodes of a graph.
            """
            return pseudo_peripheral_nodes(csr_matrix(adjacency_matrix(self)))

except ImportError:  # pragma: no cover
    Graph = None
