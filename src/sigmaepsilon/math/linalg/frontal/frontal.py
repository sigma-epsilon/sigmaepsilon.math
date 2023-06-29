# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import LinAlgError

from .proc import frontal_sym_bulk_uniform
from .postproc import backsub_fr
from .path import order_to_path


def frsolve(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray = None,
    presc_val: np.ndarray = None,
    topology: np.ndarray = None,
    epath: np.ndarray = None,
    eorder=None,
):
    try:
        nEQ = len(B)
        if len(B.shape) == 1:
            B = B.reshape((nEQ, 1))
        if epath is None:
            if eorder is not None:
                epath = order_to_path(eorder)
            else:
                epath = np.arange(len(A))
        pre = presc_val is not None
        if not pre:
            presc_bool = np.zeros((nEQ,), dtype=int)
            presc_val = np.zeros((nEQ,), dtype=float)
        lhs, rhs, eqpath, glob_to_front, glob_to_width = frontal_sym_bulk_uniform(
            A, topology, B, presc_bool, presc_val, epath
        )
        res = backsub_fr(
            lhs, rhs, presc_bool, presc_val, eqpath, glob_to_front, glob_to_width
        )
        if pre:
            return res
        else:
            return res[0]
    except Exception:
        raise LinAlgError("The matrix is singular!")


if __name__ == "__main__":
    pass
    """
    from polydata.mechanics.fem import Uniform3D
    from polydata.mechanics.material.hooke import Hooke
    from dewloosh.math.linalg import solve
    from polydata.core import NestedDefaultDict
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from time import time

    solutions = NestedDefaultDict()
    solvers = ['numpy']
    for solver in solvers:

        # geometry
        Lx, Ly, Lz = shape = (100, 100, 100)

        # mesh control
        nx, ny, nz = size = 3 * (11,)

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
        nTOTV = len(points) * 3
        if solver == 'frontal':
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
            epath = np.arange(len(cells))

            # solution
            A = K
            B = RHS.reshape((nTOTV, 1))
            topology = Ktopo
            t1 = time()
            U, R = frsolve(A, B, presc_bool, presc_val, topology, epath)
            t2 = time()
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)
        elif solver == 'numpy':
            K = assembly._K

            # essential boundary conditions
            spind = np.concatenate([points[ID].gdofs
                                    for ID in penalties.keys()])
            spdata = np.concatenate([v for v in penalties.values()])
            K += coo_matrix((spdata, (spind, spind)), shape=(nTOTV, nTOTV),
                            dtype=np.float64)

            # natural boundary conditions
            RHS = np.zeros((len(points), 3), dtype=np.float64)
            for nID, f in loads.items():
                RHS[nID] = f

            # solution
            CSR = K.tocsr()
            f = RHS.flatten()
            t1 = time()
            u = spsolve(CSR, f)
            t2 = time()
            assembly.point_data['dof'] = u.reshape(RHS.shape)
            assembly.point_data['res'] = \
                np.reshape(f - np.matmul(K.todense(), u), RHS.shape)
        elif solver == 'Gauss-Jordan':
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
            t1 = time()
            U, R = solve(A, B, presc_bool, presc_val, method='Gauss-Jordan')
            t2 = time()
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)
        elif solver == 'Jordan':
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
            t1 = time()
            U, R = solve(A, B, presc_bool, presc_val, method='Jordan')
            t2 = time()
            assembly.point_data['dof'] = U.reshape(RHS.shape)
            assembly.point_data['res'] = R.reshape(RHS.shape)

        solutions[solver]['umin'] = np.min(assembly.point_data['dof'])
        solutions[solver]['umax'] = np.max(assembly.point_data['dof'])
        solutions[solver]['u[0]'] = assembly.point_data['dof'][0]
        solutions[solver]['time'] = (t2 - t1)

    print(solutions['frontal']['umax'])
    print(solutions['numpy']['umax'])
    print(solutions['Gauss-Jordan']['umax'])
    print('\n')
    print(solutions['frontal']['time'])
    print(solutions['numpy']['time'])
    print(solutions['Gauss-Jordan']['time'])
    """
