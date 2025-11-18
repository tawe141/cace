#!/usr/bin/env python3
"""
Profile the CACE representation with torch.profiler and emit Chrome trace JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from torch.profiler import ProfilerActivity, profile

from cace.data import AtomicData
from cace.modules import BesselRBF, CosineCutoff
from cace.representations.cace_representation import Cace
import cace.tools.torch_geometric.dataloader as DataLoader

# Default parameters for CACE representation
CUTOFF = 4.0
MAX_L = 3
MAX_NU = 4
N_ATOM_BASIS = 32
N_RBF = 8
NUM_MESSAGE_PASSING = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the CACE representation on CPU and GPU (if available) "
            "and export Chrome trace JSON files."
        )
    )
    parser.add_argument(
        "structure",
        nargs="?",
        help="Path to an input structure readable by ASE. "
        "If omitted, a small H2O molecule is used.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where trace files are written (default: current directory).",
    )
    parser.add_argument(
        "--trace-prefix",
        default="cace_profile",
        help="Prefix for trace filenames (default: cace_profile).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warm-up steps before profiling (default: 100).",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=10,
        help="Number of steps captured by the profiler (default: 10).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=CUTOFF,
        help=f"Cutoff radius used by the representation (default: {CUTOFF}).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the CACE module with torch.compile before profiling.",
    )
    return parser.parse_args(argv)


def create_test_structure() -> Atoms:
    """Create a simple water molecule for testing."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.9572, 0.0, 0.0],  # H
            [-0.24, 0.9266, 0.0],  # H
        ]
    )
    atomic_numbers = [8, 1, 1]  # O, H, H
    return Atoms(positions=positions, numbers=atomic_numbers)


def load_structure(path: Optional[str]) -> Atoms:
    if path is None:
        print("Using default test structure (water molecule)")
        return create_test_structure()

    print(f"Reading structure from: {path}")
    atoms_list = read(path, ":")
    if not atoms_list:
        raise RuntimeError(f"No structures found in {path}")
    return atoms_list[0]


def setup_cace_representation(zs, cutoff: float, device: str = "cpu") -> Cace:
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

    cace_repr.eval()
    return cace_repr


def prepare_batch(atoms: Atoms, cutoff: float, device: str):
    dataset = [AtomicData.from_atoms(atoms, cutoff=cutoff)]
    data_loader = DataLoader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(data_loader)).to(device)
    return batch.to_dict()


def run_warmup(model: torch.nn.Module, data_dict, device: str, steps: int) -> None:
    with torch.no_grad():
        for _ in range(steps):
            model(data_dict)
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_profiler(
    model: torch.nn.Module,
    data_dict,
    device: str,
    trace_path: Path,
    profile_steps: int,
) -> None:
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(profile_steps):
                model(data_dict)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace written to {trace_path}")


def profile_device(
    device: str,
    atoms: Atoms,
    cutoff: float,
    warmup_steps: int,
    profile_steps: int,
    trace_path: Path,
    use_compile: bool,
) -> None:
    print(f"\nProfiling on device: {device}")
    zs = sorted(set(atoms.get_atomic_numbers()))
    model = setup_cace_representation(zs, cutoff=cutoff, device=device).to(device)

    if use_compile:
        model = torch.compile(model)

    data_dict = prepare_batch(atoms, cutoff=cutoff, device=device)

    print(f"Running {warmup_steps} warm-up steps...")
    run_warmup(model, data_dict, device=device, steps=warmup_steps)

    print(f"Collecting profiler trace for {profile_steps} steps...")
    run_profiler(
        model,
        data_dict,
        device=device,
        trace_path=trace_path,
        profile_steps=profile_steps,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    atoms = load_structure(args.structure)
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("CUDA not available, GPU profiling will be skipped.", file=sys.stderr)

    output_dir = Path(args.output_dir)

    for device in devices:
        suffix = "gpu" if device.startswith("cuda") else "cpu"
        trace_name = f"{args.trace_prefix}_{suffix}.json"
        trace_path = output_dir / trace_name
        profile_device(
            device=device,
            atoms=atoms,
            cutoff=args.cutoff,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            trace_path=trace_path,
            use_compile=args.compile,
        )

    print("\nProfiling completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
