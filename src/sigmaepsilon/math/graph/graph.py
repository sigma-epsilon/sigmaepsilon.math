import numpy as np
from numba import jit
from numba.types import int64, Array
from numba.typed import Dict

from ..linalg.sparse import csr_matrix

__all__ = ["Graph", "rooted_level_structure", "pseudo_peripheral_nodes"]

try:
    import networkx as ntx

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

        >>> from sigmaepsilon.math.topology import Graph
        >>> import networkx as nx
        >>> grid = nx.grid_2d_graph(5, 5)  # 5x5 grid
        >>> G = Graph(grid)
        """

        def adjacency_matrix(self, *args, to_csr: bool = False, **kwargs):
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
            >>> from sigmaepsilon.math.topology import Graph
            >>> G = Graph([(1, 1)])
            >>> A = G.adjacency_matrix()
            >>> print(A.todense())
            [[1]]

            """
            adj = adjacency_matrix(self, *args, **kwargs)
            return csr_matrix(adj) if to_csr else adj

        def rooted_level_structure(self, root=0):
            """
            Returns the rooted level structure (RLS) of the graph.

            The call is forwarded to `rooted_level_structure`, go there
            to read about the possible arguments.

            See Also
            --------
            :func:`rooted_level_structure`

            """
            return rooted_level_structure(csr_matrix(adjacency_matrix(self)), root)

        def pseudo_peripheral_nodes(self):
            """
            Returns the indices of nodes that are possible candidates
            for being peripheral nodes of a graph.

            """
            return pseudo_peripheral_nodes(csr_matrix(adjacency_matrix(self)))

except:  # pragma: no cover
    Graph = None

int64A = Array(int64, 1, "C")


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def rooted_level_structure(adj: csr_matrix, root: int = 0) -> Dict:
    """
    Turns a sparse adjacency matrix into a rooted level structure.

    Parameters
    ----------
    adj : csr_matrix
        Adjacency matrix in CSR format.
    root : int, Optional
        Index of the root node. Default is 0.

    Returns
    -------
    dict
        A `numba` dictionary <int[:] : int[:, :]>, where the keys
        refer to different levels, and the values are the indices
        of nodes on that level.
    """
    nN = len(adj.indptr) - 1
    rls = Dict.empty(
        key_type=int64,
        value_type=int64A,
    )
    level = 0
    rls[level] = np.array([root], dtype=np.int64)
    nodes = np.zeros(nN, dtype=np.int64)
    nodes[root] = 1
    levelset = np.zeros(nN, dtype=np.int64)
    nE = 1
    while nE < nN:
        levelset[:] = 0
        for node in rls[level]:
            neighbours = adj.irow(node)
            levelset[neighbours] = 1
        for iN in range(nN):
            if nodes[iN] == 1:
                levelset[iN] = 0
        level += 1
        rls[level] = np.where(levelset == 1)[0]
        nE += len(rls[level])
        for iN in range(nN):
            if levelset[iN] == 1:
                nodes[iN] = 1
    return rls


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def pseudo_peripheral_nodes(adj: csr_matrix) -> np.ndarray:
    """
    Returns the indices of nodes that are possible candidates
    for being peripheral nodes of a graph.

    Parameters
    ----------
    adj : csr_matrix
        Adjacency matrix in CSR format.

    Returns
    -------
    numpy.ndarray
        Integer array of nodal indices.
    """

    def length_width(RLS):
        length = len(RLS)
        width = 0
        for i in range(length):
            width = max(width, len(RLS[i]))
        return length, width

    RLS = rooted_level_structure(adj, root=0)
    length, width = length_width(RLS)
    while True:
        nodes = RLS[len(RLS) - 1]
        found = False
        for _, node in enumerate(nodes):
            iRLS = rooted_level_structure(adj, root=node)
            iL, iW = length_width(iRLS)
            if (iL > length) or (iL == length and iW < width):
                RLS = iRLS
                length = iL
                width = iW
                found = True
        if not found:
            nR = len(RLS[len(RLS) - 1]) + 1
            res = np.zeros(nR, dtype=np.int64)
            res[:-1] = RLS[len(RLS) - 1]
            res[-1] = RLS[0][0]
            return res
