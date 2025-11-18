"""
Triton kernel for accelerating the CACE symmetrizer computation.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - import guard for CPU-only environments
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is required for Symmetrizer_Triton but is not available. "
            "Install Triton or run on a CUDA-enabled environment."
        )


@triton.jit
def _symmetrizer_order_kernel(
    node_ptr,
    out_ptr,
    indices_ptr,
    pref_ptr,
    slot_offsets_ptr,
    n_rows,
    n_l,
    n_combos,
    n_slots,
    order_offset,
    stride_node_row,
    stride_out_row,
    block_rows: tl.constexpr,
    block_combos: tl.constexpr,
    nu_terms: tl.constexpr,
    max_iters: tl.constexpr,
):
    """
    node_ptr: flattened tensor [n_rows, n_l]
    out_ptr: flattened tensor [n_rows, total_slots]
    indices_ptr: [n_combos, NU]
    pref_ptr: [n_combos]
    slots_ptr: [n_combos]
    """
    pid_row = tl.program_id(0)
    pid_slot = tl.program_id(1)

    row_ids = pid_row * block_rows + tl.arange(0, block_rows)
    row_mask = row_ids < n_rows
    row_offsets_node = row_ids * stride_node_row
    row_offsets_out = row_ids * stride_out_row

    slot_start = tl.load(slot_offsets_ptr + pid_slot)
    slot_end = tl.load(slot_offsets_ptr + pid_slot + 1)
    combo_base = slot_start

    node_type = node_ptr.dtype.element_ty
    acc = tl.zeros((block_rows,), dtype=node_type)

    for _ in tl.static_range(0, max_iters):
        combo_idx = combo_base + tl.arange(0, block_combos)
        combo_mask = combo_idx < slot_end
        pref = tl.load(pref_ptr + combo_idx, mask=combo_mask, other=0.0)

        values = tl.full((block_rows, block_combos), 1.0, dtype=node_type)
        for nu in range(nu_terms):
            angular_index = tl.load(
                indices_ptr + combo_idx * nu_terms + nu,
                mask=combo_mask,
                other=0,
            )
            node_vals = tl.load(
                node_ptr + row_offsets_node[:, None] + angular_index[None, :],
                mask=row_mask[:, None] & combo_mask[None, :],
                other=1.0,
            )
            values *= node_vals

        values *= pref[None, :]
        acc += tl.sum(values, axis=1)
        combo_base += block_combos

    out_ptrs = out_ptr + row_offsets_out + (order_offset + pid_slot)
    tl.store(out_ptrs, acc, mask=row_mask)


def run_symmetrizer_order_kernel(
    node_flat: torch.Tensor,
    out_flat: torch.Tensor,
    indices: torch.Tensor,
    prefactors: torch.Tensor,
    slot_offsets: torch.Tensor,
    order_offset: int,
    *,
    block_rows: int = 128,
    block_combos: int = 32,
    max_iters: int = 1,
) -> None:
    """
    Launch the Triton kernel for a given symmetry order.

    Args:
        node_flat: Flattened tensor [num_rows, n_l].
        out_flat: Flattened destination tensor [num_rows, total_slots].
        indices: Combination indices tensor [n_combos, nu].
        prefactors: Prefactors tensor [n_combos].
        order_offset: Offset in the angular dimension to write into.
        block_rows: Triton block size over rows.
        block_combos: Triton block size over combo dimension.
    """
    _require_triton()

    if node_flat.ndim != 2 or out_flat.ndim != 2:
        raise ValueError("node_flat and out_flat must be 2D tensors")

    n_rows, n_l = node_flat.shape
    n_combos, nu = indices.shape
    if n_combos == 0:
        return

    assert prefactors.shape[0] == n_combos
    n_slots = slot_offsets.shape[0] - 1
    grid = (
        triton.cdiv(n_rows, block_rows),
        n_slots,
    )

    stride_node_row = node_flat.stride(0)
    stride_out_row = out_flat.stride(0)

    _symmetrizer_order_kernel[grid](
        node_flat,
        out_flat,
        indices,
        prefactors,
        slot_offsets,
        n_rows,
        n_l,
        n_combos,
        n_slots,
        order_offset,
        stride_node_row,
        stride_out_row,
        block_rows=block_rows,
        block_combos=block_combos,
        nu_terms=nu,
        max_iters=max_iters,
    )
