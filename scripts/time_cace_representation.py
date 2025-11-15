#!/usr/bin/env python3
"""
Script to measure the walltime of computing the CACE representation.
Can accept an input file or use a simple test structure.
"""

import sys
import time
import torch
import numpy as np
from ase import Atoms
from ase.io import read

from cace.data import AtomicData
from cace.representations.cace_representation import Cace
from cace.modules import CosineCutoff, BesselRBF
import cace.tools.torch_geometric.dataloader as DataLoader

# Default parameters for CACE representation
CUTOFF = 4.0
MAX_L = 3
MAX_NU = 4
N_ATOM_BASIS = 32
N_RBF = 8
NUM_MESSAGE_PASSING = 0


def create_test_structure():
    """Create a simple water molecule for testing."""
    positions = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.9572, 0.0, 0.0],   # H
        [-0.24, 0.9266, 0.0], # H
    ])
    atomic_numbers = [8, 1, 1]  # O, H, H
    return Atoms(positions=positions, numbers=atomic_numbers)


def setup_cace_representation(zs, cutoff, device='cpu'):
    """Set up the CACE representation model."""
    radial_basis = BesselRBF(cutoff=cutoff, n_rbf=N_RBF, trainable=False)
    cutoff_fn = CosineCutoff(cutoff=cutoff)
    
    cace_repr = Cace(
        zs=zs,
        n_atom_basis=N_ATOM_BASIS,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        max_l=MAX_L,
        max_nu=MAX_NU,
        num_message_passing=NUM_MESSAGE_PASSING,
        device=torch.device(device),
        timeit=False,
    )
    
    cace_repr.eval()  # Set to evaluation mode
    return cace_repr


def main():
    """Main function to time CACE representation computation."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Read structure from file
        input_file = sys.argv[1]
        print(f"Reading structure from: {input_file}")
        try:
            atoms_list = read(input_file, ":")
            if len(atoms_list) == 0:
                print("Error: No structures found in file")
                return
            atoms = atoms_list[0]  # Use first structure
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Use default test structure
        print("Using default test structure (water molecule)")
        atoms = create_test_structure()
    
    # Get unique atomic numbers
    zs = sorted(list(set(atoms.get_atomic_numbers())))
    print(f"Atomic numbers found: {zs}")
    print(f"Number of atoms: {len(atoms)}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set up CACE representation
    print("Setting up CACE representation...")
    cace_repr = setup_cace_representation(zs, CUTOFF, device=device)
    cace_repr = cace_repr.to(device)
    cace_repr = torch.compile(cace_repr)
    
    # Prepare data using DataLoader (consistent with other scripts)
    dataset = [AtomicData.from_atoms(atoms, cutoff=CUTOFF)]
    data_loader = DataLoader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    
    # Get the batch
    batch = next(iter(data_loader)).to(device)
    
    # Convert to dictionary format expected by CACE
    data_dict = batch.to_dict()
    
    # Warm-up run (to avoid initialization overhead)
    print("Warming up...")
    with torch.no_grad():
        for _ in range(100):
            _ = cace_repr(data_dict)
    
    # Synchronize if using GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time the computation
    print("Timing CACE representation computation...")
    num_runs = 1000  # Run multiple times for better statistics
    times = []
    
    for i in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            result = cace_repr(data_dict)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)
    
    # Print results
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n" + "="*60)
    print("CACE Representation Computation Timing Results")
    print("="*60)
    print(f"Number of runs: {num_runs}")
    print(f"Average walltime: {avg_time*1000:.4f} ms")
    print(f"Standard deviation: {std_time*1000:.4f} ms")
    print(f"Minimum walltime: {min_time*1000:.4f} ms")
    print(f"Maximum walltime: {max_time*1000:.4f} ms")
    print(f"Average walltime: {avg_time:.6f} seconds")
    print("="*60)
    
    # Print output shape for verification
    if 'node_feats' in result:
        print(f"\nOutput node_feats shape: {result['node_feats'].shape}")
    print()


if __name__ == "__main__":
    main()

