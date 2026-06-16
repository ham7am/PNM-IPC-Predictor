import openpnm as op
import numpy as np
import matplotlib.pyplot as plt

def make_pn_run_pn(Np):
    pn = op.network.Demo(shape=[Np, Np, 1], spacing=1e-4)
    air_test = op.phase.Air(network=pn)
    air_test['pore.contact_angle'] = 120
    air_test['pore.surface_tension'] = 0.072
    air_test.add_model(propname='throat.entry_pressure', model=op.models.physics.capillary_pressure.washburn,
                       surface_tension='throat.surface_tension',
                       contact_angle='throat.contact_angle',
                       diameter='throat.diameter') 
    ip_test = op.algorithms.InvasionPercolation(network=pn, phase=air_test)
    ip_test.set_inlet_BC(pores=pn.pores('left'))
    ip_test.run()
    data_test = ip_test.pc_curve()
    
    plt.figure()
    plt.plot(data_test.pc, data_test.snwp, 'b-')
    plt.xlabel('Capillary Pressure (Pa)')
    plt.ylabel('Non-wetting Phase Saturation')
    plt.title('Invasion Percolation Curve')
    plt.grid(True)
    plt.show()
    return pn
    
def split_network_matrix(matrix, Np):
    """
    Split a merged network matrix into separate pore and throat matrices.
    
    Parameters
    ----------
    matrix : np.ndarray, shape (2*Np-1, 2*Np-1)
        Merged matrix with pores at even-even and throats at odd-even/even-odd
    Np : int
        Number of pores along one side
    
    Returns
    -------
    pore_matrix : np.ndarray, shape (2*Np-1, 2*Np-1)
        Matrix with only pore diameters at even-even positions, rest zeros
    throat_matrix : np.ndarray, shape (2*Np-1, 2*Np-1)
        Matrix with only throat diameters at odd-even/even-odd positions, rest zeros
    """
    size = 2 * Np - 1
    pore_matrix = np.zeros((size, size))
    throat_matrix = np.zeros((size, size))
    
    # Extract pores (even-even positions)
    for i in range(Np):
        for j in range(Np):
            pore_matrix[2*i, 2*j] = matrix[2*i, 2*j]
    
    # Extract throats (odd-even and even-odd positions)
    for i in range(size):
        for j in range(size):
            if (i % 2) != (j % 2):  # One even, one odd
                throat_matrix[i, j] = matrix[i, j]
    
    return pore_matrix, throat_matrix

def build_network_object(matrices, Np: int, spacing=1e-4):
    """
    Reconstruct an OpenPNM network object from encoded matrices with spacing.
    
    Takes the inverse of build_network_matrix to recreate a network from its
    matrix representation. Applies spacing to coordinates for proper throat length.
    
    Parameters
    ----------
    matrices : tuple/list of (np.ndarray, np.ndarray)
        First matrix contains pore diameters at even-even positions.
        Second matrix contains throat diameters at even-odd/odd-even positions.
        Odd-odd positions are ignored.
    Np : int
        Number of pores along one side (3 for a 3×3 network).
    spacing : float, optional
        Spacing between pores. Default is 1e-4.
    
    Returns
    -------
    pn : openpnm.network.Network
        Network object with pore and throat diameters assigned.
    """
    if isinstance(matrices, (list, tuple)) and len(matrices) == 2:
        pore_matrix, throat_matrix = matrices
    elif isinstance(matrices, np.ndarray):
        # Single merged matrix - extract pore and throat from same matrix
        pore_matrix = matrices
        throat_matrix = matrices
    else:
        raise ValueError("Expected tuple/list of 2 matrices or single merged matrix")
    
    # Extract pore diameters from even-even positions
    pore_diameters = []
    for i in range(Np):
        for j in range(Np):
            pore_diameters.append(pore_matrix[2*i, 2*j])
    
    # Create pore coordinates with spacing applied
    pore_coords = []
    for i in range(Np):
        for j in range(Np):
            pore_coords.append([i * spacing, j * spacing, 0])
    
    # Build throat connectivity (standard 3D lattice, confined to 2D)
    throat_conns = []
    throat_diameters = []
    
    for i in range(Np):
        for j in range(Np):
            p_idx = i * Np + j
            
            # Right neighbor (moves to j+1)
            if j < Np - 1:
                p_next = i * Np + (j + 1)
                throat_conns.append([p_idx, p_next])
                throat_diameters.append(throat_matrix[2*i, 2*j + 1])
            
            # Bottom neighbor (moves to i+1)
            if i < Np - 1:
                p_next = (i + 1) * Np + j
                throat_conns.append([p_idx, p_next])
                throat_diameters.append(throat_matrix[2*i + 1, 2*j])
    
    # Create OpenPNM network
    pn = op.network.Cubic(shape=[Np, Np, 1], spacing=spacing)
    
    mods = op.models.collections.geometry.spheres_and_cylinders
    pn.add_model_collection(mods)
    pn['pore.diameter'] = np.array(pore_diameters)
    pn['throat.conns'] = np.array(throat_conns)
    pn['throat.diameter'] = np.array(throat_diameters)
    pn['pore.coords'] = np.array(pore_coords)
    pn.regenerate_models(exclude=['pore.diameter', 'throat.diameter'])
    
    return pn

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

def run_pn(pn, Np: int):
    air_test = op.phase.Air(network=pn)
    air_test['pore.contact_angle'] = 120
    air_test['pore.surface_tension'] = 0.072
    air_test.add_model(propname='throat.entry_pressure', model=op.models.physics.capillary_pressure.washburn,
                       surface_tension='throat.surface_tension',
                       contact_angle='throat.contact_angle',
                       diameter='throat.diameter') 
    ip_test = op.algorithms.InvasionPercolation(network=pn, phase=air_test)
    ip_test.set_inlet_BC(pores=pn.pores('left'))
    ip_test.run()
    data_test = ip_test.pc_curve()
    
    plt.figure()
    plt.plot(data_test.pc, data_test.snwp, 'r-')
    plt.xlabel('Capillary Pressure (Pa)')
    plt.ylabel('Non-wetting Phase Saturation')
    plt.title('Invasion Percolation Curve (from Network Matrix)')
    plt.grid(True)
    plt.show()
    
def run_pn2(pn, Np: int):
    mat = build_network_matrix(pn, Np)
    pm, tm = split_network_matrix(mat, Np)
    matrices = [pm, tm]
    pn_rebuild = build_network_object(matrices, Np)
    
    air_test = op.phase.Air(network=pn_rebuild)
    air_test['pore.contact_angle'] = 120
    air_test['pore.surface_tension'] = 0.072
    air_test.add_model(propname='throat.entry_pressure', model=op.models.physics.capillary_pressure.washburn,
                       surface_tension='throat.surface_tension',
                       contact_angle='throat.contact_angle',
                       diameter='throat.diameter') 
    ip_test = op.algorithms.InvasionPercolation(network=pn_rebuild, phase=air_test)
    ip_test.set_inlet_BC(pores=pn_rebuild.pores('left'))
    ip_test.run()
    data_test = ip_test.pc_curve()
    
    plt.figure()
    plt.plot(data_test.pc, data_test.snwp, 'r-')
    plt.xlabel('Capillary Pressure (Pa)')
    plt.ylabel('Non-wetting Phase Saturation')
    plt.title('Invasion Percolation Curve (from Network Matrix)')
    plt.grid(True)
    plt.show()