# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

from .frutils import max_frontwidth_bulk, flatsize_sym, flatind_sym
from .topo import signed_topo_bulk


@jit(nopython=True, nogil=True, cache=True)
def frontal_sym_bulk_uniform(
    A: np.ndarray,
    topology: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray,
    presc_val: np.ndarray,
    epath: np.ndarray,
):
    nEQ, nRHS = B.shape
    nE, nXE = topology.shape
    topology += 1
    topo_sig = signed_topo_bulk(topology, epath)
    maxwidth = max_frontwidth_bulk(topo_sig, epath)
    A_front = np.zeros(flatsize_sym(maxwidth), dtype=A.dtype)
    B_front = np.zeros((maxwidth, nRHS), dtype=A.dtype)
    front_to_glob = np.zeros(maxwidth, dtype=topology.dtype)
    loc_to_front = np.zeros(nXE, dtype=topology.dtype)
    lhs = np.zeros((nEQ, maxwidth), dtype=A.dtype)
    rhs = np.zeros((nEQ, nRHS), dtype=A.dtype)
    glob_to_width = np.zeros(nEQ, dtype=topology.dtype)
    glob_to_front = np.zeros(nEQ, dtype=topology.dtype)
    eqpath = np.zeros(nEQ, dtype=topology.dtype)
    frontwidth = 0

    # ASSEMBLY & ELIMINATION
    iEQ = 0
    for iE in epath:
        # find front indices for local variables
        for iXE in range(nXE):
            gind = topology[iE, iXE]
            frontindex = -1
            for ifront in range(frontwidth):
                if front_to_glob[ifront] == gind:
                    frontindex = ifront
                    break
            if frontindex < 0:
                for ifront in range(frontwidth):
                    if front_to_glob[ifront] == 0:
                        frontindex = ifront
                        break
            if frontindex < 0:
                frontindex = frontwidth
                frontwidth += 1
            front_to_glob[frontindex] = gind
            loc_to_front[iXE] = frontindex

        # local assembly
        nElim = 0
        for iXE in range(nXE):
            ifront = loc_to_front[iXE]
            for jXE in range(iXE, nXE):
                jfront = loc_to_front[jXE]
                index = flatind_sym(ifront, jfront)
                A_front[index] += A[iE, iXE, jXE]
            gind = -topo_sig[iE, iXE]
            if gind > 0:
                # because RHS is assumed to be present as a full array,
                # assembly happends only the last time the variable is
                # accessed
                B_front[ifront] += B[gind - 1]
                nElim += 1

        # local elimination
        cElim = 0
        for iXE in range(nXE):
            # check if variable needs to be handled
            if cElim == nElim:
                break
            ifront = loc_to_front[iXE]
            gind = -topo_sig[iE, iXE]
            if gind < 0:
                continue
            else:
                gind -= 1

            # store equation
            for jfront in range(frontwidth):
                kfront = flatind_sym(ifront, jfront)
                lhs[gind, jfront] = A_front[kfront]
                A_front[kfront] = 0.0
            rhs[gind] = B_front[ifront]
            B_front[ifront] = 0.0

            # eliminate from front
            pivot = lhs[gind, ifront]
            lhs[gind, ifront] = 0.0
            if presc_bool[gind] == True:
                for jfront in range(frontwidth):
                    B_front[jfront] -= lhs[gind, jfront] * presc_val[gind]
            else:
                if abs(pivot) < 1e-12:
                    raise Exception("The matrix is singular!")
                for jfront in range(frontwidth):
                    factor = lhs[gind, jfront] / pivot
                    if abs(factor) < 1e-12:
                        continue
                    for kfront in range(jfront, frontwidth):
                        lfront = flatind_sym(jfront, kfront)
                        A_front[lfront] -= factor * lhs[gind, kfront]
                    B_front[jfront] -= factor * rhs[gind]
            lhs[gind, ifront] = pivot
            front_to_glob[ifront] = 0

            # store data for backsubstitution
            glob_to_width[gind] = frontwidth
            glob_to_front[gind] = ifront
            eqpath[iEQ] = gind
            cElim += 1
            iEQ += 1

        # decrease frontwidth if possible
        for ifront in range(frontwidth - 1, -1, -1):
            if front_to_glob[ifront] != 0:
                break
            frontwidth -= 1
    topology -= 1
    return lhs, rhs, eqpath, glob_to_front, glob_to_width


if __name__ == "__main__":
    pass
    """
    from polydata.mechanics.fem import Uniform3D
    from polydata.mechanics.material.hooke import Hooke
    from dewloosh.math.linalg.frontal.path import optimal_elimination_path
    from dewloosh.math.linalg import solve
    from dewloosh.math.linalg.frontal.postproc import backsub_fr
    from polydata.core import NestedDefaultDict
    import numpy as np

    solutions = NestedDefaultDict()
    solvers = ['frontal', 'numpy', 'Gauss-Jordan']
    for solver in solvers:

        # geometry
        Lx, Ly, Lz = shape = (100, 100, 100)

        # mesh control
        nx, ny, nz = size = (5, 5, 5)

        # assembly
        assembly = Uniform3D(shape, size)
        cells = assembly.cells()
        points = assembly.points()
        coord = np.array(points)

        # material
        model = Hooke(E=2000, NU=0.3)
        for c in cells:
            c.model = model

        # essential boundary conditions
        condZ = (abs(coord[:, 2] - Lz/2) < 0.001)
        condX = (coord[:, 0] < 0.001) | (coord[:, 0] > Lx-0.001)
        condY = (coord[:, 1] < 0.001) | (coord[:, 1] > Ly-0.001)
        cond = (condX | condY) & condZ
        nIDs = np.where(cond)[0]
        if solver == 'numpy':
            p = np.array([1e12, 1e12, 1e12], dtype=np.float64)
            penalties = {nID: p for nID in nIDs}
        else:
            disps = {}
            for nID in nIDs:
                for idof in range(3):
                    disps[nID * 3 + idof] = 0.0

        # natural boundary conditions
        case = 0
        if case == 0:
            nIDs = np.where(condZ)[0]
            fz = (-10/10000)*Lx*Ly/len(nIDs)
            fp = np.array([0, 0, fz], dtype=np.float64)
            loads = {nID: fp for nID in nIDs}
        elif case == 1:
            condZ = (abs(coord[:, 2] - Lz/2) < 0.001)
            condX = (abs(coord[:, 0] - Lx/2) < 0.001)
            condY = (abs(coord[:, 1] - Ly/2) < 0.001)
            cond = condX & condY & condZ
            nIDs = np.where(cond)[0]
            loads = {nIDs[0]: np.array([0., 0., -10.0])}

        # preprocessor
        if solver == 'frontal':
            assembly.preprocess(solver=solver)
        elif solver == 'numpy':
            assembly.preprocess(solver=solver)
        elif solver == 'Gauss-Jordan':
            assembly.preprocess(solver='numpy')
        elif solver == 'Jordan':
            assembly.preprocess(solver='numpy')

        # solve
        if solver == 'frontal':
            nTOTV = len(points) * 3

            # essential boundary conditions
            presc_bool = np.zeros(nTOTV, dtype=np.int64)
            presc_val = np.zeros(nTOTV, dtype=np.float64)
            for dID, dval in disps.items():
                presc_val[dID] = dval
                presc_bool[dID] = 1

            # natural boundary conditions
            RHS = np.zeros((len(points), 3), dtype=np.float64)
            for nID, f in loads.items():
                RHS[nID] = f

            # stiffness
            K, Ktopo = assembly.stiffness_matrix_frontal()

            # connectivity and topology
            adj_csr = assembly.adjacency_matrix(sparse=True, mode='node')
            topo_csr = assembly.connectivity_matrix(sparse=True,
                                                    mode='node')
            epath = optimal_elimination_path(adj_csr, topo_csr)
            #epath = np.arange(len(cells))

            # solution
            A = K
            B = RHS.reshape((nTOTV, 1))
            topology = Ktopo
            lhs, rhs, eqpath, glob_to_front, glob_to_width = \
                frontal_sym_bulk_uniform(A, topology, B,
                                         presc_bool, presc_val, epath)
            U, R = backsub_fr(lhs, rhs, presc_bool, presc_val, eqpath,
                              glob_to_front, glob_to_width)
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)
        elif solver == 'numpy':
            assembly.process(nodal_penalties=penalties, nodal_loads=loads,
                             solver=solver)
        elif solver == 'Gauss-Jordan':
            nTOTV = len(points) * 3

            # essential boundary conditions
            presc_bool = np.zeros(nTOTV, dtype=np.int64)
            presc_val = np.zeros(nTOTV, dtype=np.float64)
            for dID, dval in disps.items():
                presc_val[dID] = dval
                presc_bool[dID] = 1

            # natural boundary conditions
            RHS = np.zeros((len(points), 3), dtype=np.float64)
            for nID, f in loads.items():
                RHS[nID] = f

            # stiffness
            A = assembly._K.todense()
            B = RHS.reshape((nTOTV, 1))
            U, R = solve(A, B, presc_bool, presc_val, method='Gauss-Jordan')
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)
        elif solver == 'Jordan':
            nTOTV = len(points) * 3

            # essential boundary conditions
            presc_bool = np.zeros(nTOTV, dtype=np.int64)
            presc_val = np.zeros(nTOTV, dtype=np.float64)
            for dID, dval in disps.items():
                presc_val[dID] = dval
                presc_bool[dID] = 1

            # natural boundary conditions
            RHS = np.zeros((len(points), 3), dtype=np.float64)
            for nID, f in loads.items():
                RHS[nID] = f

            # stiffness
            A = assembly._K.todense()
            B = RHS.reshape((nTOTV, 1))
            U, R = solve(A, B, presc_bool, presc_val, method='Jordan')
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)

        solutions[solver]['min'] = np.min(assembly.point_data['dof'])
        solutions[solver]['max'] = np.max(assembly.point_data['dof'])
        solutions[solver]['u[0]'] = assembly.point_data['dof'][0]

    print(solutions['frontal']['max'])
    print(solutions['numpy']['max'])
    print(solutions['Gauss-Jordan']['max'])
    """
