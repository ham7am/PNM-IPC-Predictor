import openpnm as op
import numpy as np
import matplotlib.pyplot as plt

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
    pn = op.network.Demo(shape=[Np, Np, 1], spacing=spacing)
    pn['pore.diameter'] = np.array(pore_diameters)
    pn['throat.conns'] = np.array(throat_conns)
    pn['throat.diameter'] = np.array(throat_diameters)
    pn['pore.coords'] = np.array(pore_coords)
    
    return pn

def build_network_from_original(matrices, Np: int, original_pn):
    """
    Reconstruct an OpenPNM network by copying all properties from original network.
    
    Extracts pore and throat diameters from matrices, but copies coordinates,
    connectivity, and other properties from the original network to ensure
    exact geometric equivalence.
    
    Parameters
    ----------
    matrices : tuple/list of (np.ndarray, np.ndarray)
        First matrix contains pore diameters at even-even positions.
        Second matrix contains throat diameters at even-odd/odd-even positions.
    Np : int
        Number of pores along one side (3 for a 3×3 network).
    original_pn : openpnm.network.Network
        Original network to copy coordinates and connectivity from.
    
    Returns
    -------
    pn : openpnm.network.Network
        Network object with all properties matching original network.
    """
    if isinstance(matrices, (list, tuple)) and len(matrices) == 2:
        pore_matrix, throat_matrix = matrices
    elif isinstance(matrices, np.ndarray):
        pore_matrix = matrices
        throat_matrix = matrices
    else:
        raise ValueError("Expected tuple/list of 2 matrices or single merged matrix")
    
    # Extract pore diameters from even-even positions
    pore_diameters = []
    for i in range(Np):
        for j in range(Np):
            pore_diameters.append(pore_matrix[2*i, 2*j])
    
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
    
    # Reorder throat diameters to match original network's connectivity
    orig_conns = original_pn['throat.conns']
    rebuilt_dict = {}
    for i, (p1, p2) in enumerate(throat_conns):
        rebuilt_dict[(p1, p2)] = throat_diameters[i]
        rebuilt_dict[(p2, p1)] = throat_diameters[i]
    
    reordered_diameters = []
    for p1, p2 in orig_conns:
        if (p1, p2) in rebuilt_dict:
            reordered_diameters.append(rebuilt_dict[(p1, p2)])
        else:
            reordered_diameters.append(rebuilt_dict[(p2, p1)])
    
    # Create new network by copying original
    pn = op.network.Demo(shape=[Np, Np, 1], spacing=1e-4)
    pn['pore.diameter'] = np.array(pore_diameters)
    pn['pore.coords'] = original_pn['pore.coords'].copy()
    pn['throat.conns'] = original_pn['throat.conns'].copy()
    pn['throat.diameter'] = np.array(reordered_diameters)
    
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

def run_sim(Np: int):
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

def run_sim_mat(pn, Np: int):
    matrices = build_network_matrix(pn, Np)
    pn_rebuild = build_network_from_original(matrices, Np, original_pn=pn)
    
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
    
def run_sim_pn(pn, Np: int):
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
    plt.title('Invasion Percolation Curve')
    plt.grid(True)
    plt.show()