import openpnm as op
import numpy as np
import random
import matplotlib.pyplot as plt


def generate_cubic_network(n, pore_size_range=(1e-6, 5e-6), 
                          throat_size_range=(0.5e-6, 3e-6), 
                          seed=None):
    """
    Generate a single nxnxn cubic network with randomly distributed pore and throat sizes.
    
    Parameters
    ----------
    n : int
        Size of the cubic network (creates an nxnxn network)
    pore_size_range : tuple, optional
        Min and max diameter for pores in meters (default: 1e-6 to 5e-6 m)
    throat_size_range : tuple, optional
        Min and max diameter for throats in meters (default: 0.5e-6 to 3e-6 m)
    seed : int, optional
        Random seed for reproducibility (default: None)
    
    Returns
    -------
    network : openpnm.network.Network
        The generated cubic network with random pore and throat sizes
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create cubic network
    network = op.network.Cubic(shape=[n, n, n])
    
    # Generate random pore diameters
    pore_diameters = np.random.uniform(
        pore_size_range[0], 
        pore_size_range[1], 
        network.Np
    )
    network['pore.diameter'] = pore_diameters
    
    # Generate random throat diameters
    throat_diameters = np.random.uniform(
        throat_size_range[0], 
        throat_size_range[1], 
        network.Nt
    )
    network['throat.diameter'] = throat_diameters
    
    return network


def generate_ipc_curve(pore_diameter, throat_diameter, n_size):
    """
    Generate a fresh cubic network, add geometry and phase models, run drainage and 
    invasion percolation simulations, and generate IPC curve.
    
    Parameters
    ----------
    pore_diameter : np.ndarray
        Pore diameters
    throat_diameter : np.ndarray
        Throat diameters
    n_size : int
        Network size (for cubic shape [n, n, n])
    
    Returns
    -------
    pc_curve : object
        IPC curve data with .pc and .snwp attributes
    """
    try:
        # Create fresh cubic network (preserves boundary labels like 'left', 'right')
        network = op.network.Cubic(shape=[n_size, n_size, n_size], spacing=1e-4)
        
        # Set diameters
        network['pore.diameter'] = pore_diameter
        network['throat.diameter'] = throat_diameter
        
        # Add geometry models
        network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        network.regenerate_models()
        
        # Create air phase
        air = op.phase.Air(network=network)
        air['throat.surface_tension'] = 0.072
        air['throat.contact_angle'] = 120
        
        # Add capillary pressure model
        air.add_model(propname='throat.entry_pressure',
                      model=op.models.physics.capillary_pressure.washburn,
                      surface_tension='throat.surface_tension',
                      contact_angle='throat.contact_angle',
                      diameter='throat.diameter')
        
        # Run drainage simulation
        drn = op.algorithms.Drainage(network=network, phase=air)
        drn.set_inlet_BC(pores=network.pores('left'))
        drn.run()
        
        # Run invasion percolation to get actual IPC curve
        ip = op.algorithms.InvasionPercolation(network=network, phase=air)
        ip.set_inlet_BC(pores=network.pores('left'))
        ip.run()
        
        # Extract IPC curve from invasion percolation
        pc_curve = ip.pc_curve()
        
        # Plot the curve
        plt.figure(figsize=(8, 6))
        plt.plot(pc_curve.pc, pc_curve.snwp, 'b-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Capillary Pressure (Pa)')
        plt.ylabel('Saturation (Snwp)')
        plt.title('IPC Curve - Invasion Percolation')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return pc_curve
        
    except Exception as e:
        print(f'Error generating IPC curve: {e}')
        return None


def network_to_tensor(pore_coords, pore_diameter, throat_conns, throat_diameter, n_size):
    """
    Convert network data to a 3D spatial tensor representation.
    
    For an nxnxn cubic network, creates a (2n-1, 2n-1, 2n-1) tensor where:
    - Pores occupy even indices: (2i, 2j, 2k)
    - Throats occupy odd indices between pores: (2i+1, 2j, 2k), (2i, 2j+1, 2k), (2i, 2j, 2k+1)
    
    Parameters
    ----------
    pore_coords : np.ndarray
        Flattened pore coordinates, reshaped to (n_pores, 3)
    pore_diameter : np.ndarray
        Pore diameters
    throat_conns : np.ndarray
        Throat connections (pairs of pore indices), reshaped to (n_throats, 2)
    throat_diameter : np.ndarray
        Throat diameters
    n_size : int
        Network size (for cubic shape [n, n, n])
    
    Returns
    -------
    tensor : np.ndarray
        3D tensor of shape (2*n_size-1, 2*n_size-1, 2*n_size-1) with pore and throat diameters
    """
    # Create output tensor
    tensor_size = 2 * n_size - 1
    tensor = np.zeros((tensor_size, tensor_size, tensor_size))
    
    # Reshape pore coordinates
    coords = pore_coords.reshape(-1, 3).astype(int)
    
    # Place pore diameters at even indices
    for pore_idx, (i, j, k) in enumerate(coords):
        tensor[2*i, 2*j, 2*k] = pore_diameter[pore_idx]
    
    # Reshape throat connections
    conns = throat_conns.reshape(-1, 2).astype(int)
    
    # Place throat diameters between pores
    for throat_idx, (pore1, pore2) in enumerate(conns):
        pos1 = coords[pore1]
        pos2 = coords[pore2]
        
        # Calculate position between pores
        mid_pos = ((2 * pos1[0] + 2 * pos2[0]) // 2,
                   (2 * pos1[1] + 2 * pos2[1]) // 2,
                   (2 * pos1[2] + 2 * pos2[2]) // 2)
        
        # Place throat diameter
        tensor[mid_pos[0], mid_pos[1], mid_pos[2]] = throat_diameter[throat_idx]
    
    return tensor


if __name__ == '__main__':
    n_samples = 10000  # Number of samples to generate
    n_size = 5  # Network size (5x5x5)
    
    # Initialize lists to collect data as numpy arrays
    pore_coords_list = []
    pore_diameter_list = []
    throat_conns_list = []
    throat_diameter_list = []
    
    for i in range(n_samples):
        network = generate_cubic_network(n_size, seed=i * random.randint(1, 10000))
        
        # Collect network data
        pore_coords_list.append(network['pore.coords'].flatten())
        pore_diameter_list.append(network['pore.diameter'].flatten())
        throat_conns_list.append(network['throat.conns'].flatten())
        throat_diameter_list.append(network['throat.diameter'].flatten())
        
        print(f'Generated network {i}')
    
    # Save networks as numpy arrays
    networks_data = {
        'pore_coords': np.array(pore_coords_list),
        'pore_diameter': np.array(pore_diameter_list),
        'throat_conns': np.array(throat_conns_list),
        'throat_diameter': np.array(throat_diameter_list),
    }
    np.savez('data/networks.npz', **networks_data)
    print(f'Saved {n_samples} networks to data/networks.npz')
    
    # Generate IPC curves
    pressure_list = []
    saturation_list = []
    
    for i in range(n_samples):
        print(f'Generating IPC curve for network {i}...')
        
        pc_curve = generate_ipc_curve(
            pore_diameter=pore_diameter_list[i],
            throat_diameter=throat_diameter_list[i],
            n_size=n_size
        )
        
        if pc_curve is not None:
            pressure_list.append(pc_curve.pc)
            saturation_list.append(pc_curve.snwp)
            print(f'IPC curve {i} done')
        else:
            print(f'IPC curve {i} failed')
    
    # Save IPC curves as numpy arrays
    ipc_data = {
        'pressure': np.array(pressure_list),
        'saturation': np.array(saturation_list),
    }
    np.savez('data/ipc.npz', **ipc_data)
    print(f'Saved {len(pressure_list)} IPC curves to data/ipc.npz')
    
    # Convert networks to tensors using numpy arrays directly
    print('\nConverting networks to spatial tensors...')
    tensors = []
    
    for idx in range(n_samples):
        tensor = network_to_tensor(
            pore_coords=networks_data['pore_coords'][idx],
            pore_diameter=networks_data['pore_diameter'][idx],
            throat_conns=networks_data['throat_conns'][idx],
            throat_diameter=networks_data['throat_diameter'][idx],
            n_size=n_size
        )
        tensors.append(tensor)
        print(f'Converted network {idx} to tensor of shape {tensor.shape}')
    
    # Save tensors as numpy array
    input_tensors = np.array(tensors)
    np.save('data/input.npy', input_tensors)
    print(f'Saved {len(tensors)} tensors to data/input.npy with shape {input_tensors.shape}')