# -*- coding: utf-8 -*-

import openpnm as op
import numpy as np
from joblib import Parallel, delayed

def build_network_matrix(pn: op.network.Network, Np: int):
    """
    Build a (2*Np-1) x (2*Np-1) matrix from a square cubic PNM network.
    Where Np is number of pores. 
    For a square/cubic network, (2*Np-1) is Np+Nt in a row/column.
    Layout in the output matrix:
      - Even row & even col  →  pore diameter
      - One even, one odd    →  throat diameter (of the connecting throat)
      - Odd row  & odd col   →  0  (no physical meaning)

    Parameters
    ----------
    pn : openpnm Network
        Must have 'pore.diameter' and 'throat.diameter' defined.
    Np : int
        Number of pores along one side (3 for a 3×3 network).

    Returns
    -------
    matrix : np.ndarray, shape (2*Np-1, 2*Np-1)
    """
    size = 2 * Np - 1
    matrix = np.zeros((size, size))

    def pore_pos(idx):
        """Pore index → (row, col) in the output matrix."""
        return 2 * (idx // Np), 2 * (idx % Np)

    # Pore diameters at even-even positions
    for p, d in enumerate(pn['pore.diameter']):
        r, c = pore_pos(p)
        matrix[r, c] = d

    # Throat diameters at the midpoint between the two connected pore positions
    for t, (p1, p2) in enumerate(pn['throat.conns']):
        r1, c1 = pore_pos(p1)
        r2, c2 = pore_pos(p2)
        matrix[(r1 + r2) // 2, (c1 + c2) // 2] = pn['throat.diameter'][t]

    # Odd-odd positions stay 0 (already zero from np.zeros)
    return matrix

def generate_data(size: int):
    """
    Generates a network object, runs drainage simulation.
    Args:
        Network size(Np) - NpxNp. Assumes square/cubic network.
    Returns:
        Network pore-throat tensor: np.ndarray
        ipc x,y points: np.ndarray
    """
    pn = op.network.Demo(shape=[size, size, 1], spacing=1e-4)
    air = op.phase.Air(network=pn)
    
    air['pore.contact_angle'] = 120
    air['pore.surface_tension'] = 0.072
    f = op.models.physics.capillary_pressure.washburn
    air.add_model(propname='throat.entry_pressure',
                  model=f,
                  surface_tension='throat.surface_tension',
                  contact_angle='throat.contact_angle',
                  diameter='throat.diameter',)
    
    ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
    ip.set_inlet_BC(pores=pn.pores('left'))
    ip.run()
    data_ip = ip.pc_curve()
    X = build_network_matrix(pn, Np=size)
    Y = np.column_stack([data_ip.pc, data_ip.snwp])
    return X,Y

if __name__ == '__main__':
    n = 500000
    X, Y = generate_data(size=10)

    Xs = np.zeros((n, *X.shape))
    Ys = np.zeros((n, *Y.shape))
    Xs[0] = X
    Ys[0] = Y

    # Generate remaining samples in parallel
    results = Parallel(n_jobs=-1, return_as='generator', verbose=100)(delayed(generate_data)(size=10) for _ in range(1, n))

    for i, (X, Y) in enumerate(results, start=1):
        Xs[i] = X
        Ys[i] = Y

    print(f"All {n} samples generated. Saving...")
    np.savez('./data/10_10_1_training_data.npz', X=Xs, Y=Ys)
    print("Training data saved to training_data.npz")