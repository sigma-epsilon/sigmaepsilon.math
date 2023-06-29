# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

from ..sparse import CSR
from ...topology.graph import pseudo_peripheral_nodes


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def optimal_node_order(adj: CSR):
    itype = adj.indptr
    nN = len(adj.indptr) - 1  # number of nodes
    roots = pseudo_peripheral_nodes(adj)
    renum = np.full(nN, -1, dtype=itype)
    order = np.arange(0, nN, dtype=itype)
    minwidth = nN

    # 0  == virgin
    # -1 == eliminated
    # n  == age
    status = np.zeros(nN, dtype=itype)

    def iseliminated(status, i):
        return status[i] < 0

    def isactive(status, i):
        return status[i] > 0

    def isvirgin(status, i):
        return status[i] == 0

    def eliminate(status, i):
        status[i] = -1

    def activate(status, i):
        status[i] = 1

    def get_neighbours_of(adj, i):
        _, inds = adj.row(i)
        return inds

    def get_actives(status):
        return np.argwhere(status > 0).flatten()

    def get_cost(adj, status, i):
        cost = -1
        neighbours = get_neighbours_of(adj, i)
        for node in neighbours:
            if isvirgin(status, node):
                cost += 1
        return cost

    def get_costs(adj, status, actives):
        # measure cost of the elimination of active nodes
        nA = len(actives)
        costs = np.full(nA, -1, dtype=np.float64)
        for iA in range(nA):
            costs[iA] = get_cost(adj, status, actives[iA])
        return costs

    def get_cheapest(adj, status, actives):
        costs = get_costs(adj, status, actives)
        cheapest = np.argwhere(costs == np.min(costs)).flatten()
        # decide draw based on time spent in the front
        if len(cheapest) > 1:
            ages = status[actives[cheapest]]
            aind = cheapest[np.argmax(ages)]
        else:
            aind = cheapest[0]
        return actives[aind]

    nN -= 1  # for direct comparison with running variable
    nR = len(roots)
    # the root of the level structure is the last item in roots
    for iR in range(nR - 1, -1, -1):
        # initialize
        status[:] = 0
        renum[:] = -1
        nextindex = 0
        width = 0

        # register starting node, mark all neighbours as active
        root = roots[iR]
        renum[root] = nextindex
        eliminate(status, root)
        neighbours = get_neighbours_of(adj, root)
        for node in neighbours:
            if isvirgin(status, node):
                activate(status, node)

        while nN > nextindex:  # eliminate until there is nothing left
            # measure current solution
            actives = get_actives(status)
            width = max(width, len(actives))
            if width >= minwidth:
                break

            # increase age of active variables
            for node in actives:
                status[node] += 1

            # get variable cheapest to eliminate
            cheapest = get_cheapest(adj, status, actives)

            # eliminate variable and read adjacent variables
            nextindex += 1
            renum[cheapest] = nextindex
            eliminate(status, cheapest)
            neighbours = get_neighbours_of(adj, cheapest)
            for node in neighbours:
                if isvirgin(status, node):
                    activate(status, node)

        # register maximum occuring width
        if np.min(renum) == 0:
            if width < minwidth:
                order[:] = renum
                minwidth = width
    return order


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def optimal_node_path(adj: CSR = None):
    return order_to_path(optimal_node_order(adj))


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def optimal_elimination_order(adj: CSR, topo: CSR):
    # renumber nodes
    norder = optimal_node_order(adj)
    for i in range(len(topo.data)):
        topo.data[i] = norder[topo.data[i]]

    # reorder elements
    nN = len(adj.indptr) - 1  # number of nodes
    nE = len(topo.indptr) - 1  # number of elements
    eorder = np.full(nE, -1, dtype=np.int64)
    cE = 0
    for iN in range(nN):
        for iE in range(nE):
            if eorder[iE] > -1:
                continue
            else:
                nodes, _ = topo.row(iE)
                for node in nodes:
                    if node == iN:
                        eorder[iE] = cE
                        cE += 1
                        break

    # undo renumbering of nodes
    npath = order_to_path(norder)
    for i in range(len(topo.data)):
        topo.data[i] = npath[topo.data[i]]
    return eorder


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def optimal_elimination_path(adj: CSR, topo: CSR):
    return order_to_path(optimal_elimination_order(adj, topo))


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def order_to_path(order: np.ndarray):
    path = np.zeros(order.shape, dtype=order.dtype)
    for i in range(len(path)):
        path[order[i]] = i
    return path


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def path_to_order(path: np.ndarray):
    order = np.zeros(path.shape, dtype=path.dtype)
    for i in range(len(order)):
        order[path[i]] = i
    return order


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def reverse_mapping(iarray: np.ndarray):
    """
    Returns the inverse of a resequencing path represented by the
    complete integer array iarray. Here, completeness means that
    the array iarray must be a shuffled copy of np.arange(len(iarray)).
    """
    return order_to_path(iarray)


if __name__ == "__main__":
    pass
    """
    from polydata.mechanics.fem import Uniform3D
    from dewloosh.math.linalg.frontal.frutils import max_frontwidth_bulk
    from dewloosh.math.linalg.frontal.topo import signed_topo_bulk
    from dewloosh.math.linalg.sparse import CSR

    #####################
    # Example 1
    # basic operations
    #####################

    # geometry
    Lx = 100.0
    Ly = 100.0
    Lz = 100

    # mesh control
    nx = 21
    ny = 21
    nz = 21

    # assembly
    assembly = Uniform3D((Lx, Ly, Lz), (nx, ny, nz))
    points = assembly.points()
    cells = assembly.cells()

    # connectivity and topology
    topo = assembly.element_node_numbering()
    adj_csr = assembly.adjacency_matrix(sparse=True)
    topo_csr = assembly.connectivity_matrix(sparse=True)
    roots = pseudo_peripheral_nodes(adj_csr)
    norder = optimal_node_order(adj_csr)
    npath = order_to_path(norder)
    epath_0 = np.arange(len(cells))
    epath = optimal_elimination_path(adj_csr, topo_csr)
    eorder = optimal_elimination_order(adj_csr, topo_csr)
    topo_sig_0 = signed_topo_bulk(topo, epath_0)
    topo_sig = signed_topo_bulk(topo, epath)
    maxwidth = max_frontwidth_bulk(topo_sig, epath)
    maxwidth_0 = max_frontwidth_bulk(topo_sig_0, epath_0)
    print(maxwidth_0, ' --> ', maxwidth)

    #####################
    # Example 2
    # Hinton, Owen : Finite Element Programming, 1977, Academic Press,
    #     p182.
    # Statement : minimum frontwidth should be 12
    #####################

    topo_signed_1 = np.array([[-1, -2, -3, 4, 5, 6, 10, 11, 12],
                              [4, 5, 6, -13, -14, -15, 7, 8, 9],
                              [-4, -5, -6, -7, -8, -9, -10, -11, -12]])
    path_1 = np.arange(3, dtype=np.int64)
    maxwidth_topo_1 = max_frontwidth_bulk(topo_signed_1, path_1)
    """
