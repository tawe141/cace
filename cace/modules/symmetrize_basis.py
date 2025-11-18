import torch
import torch.nn as nn
import numpy as np
from opt_einsum import contract, shared_intermediates
import warnings
import math
from .angular_tools import (
    find_combo_vectors_nu2,
    find_combo_vectors_nu3,
    find_combo_vectors_nu4
    )
from .kernels.symmetrizer import HAS_TRITON, run_symmetrizer_order_kernel

__all__ = ['Symmetrizer', 'Symmetrizer_Vectorized', 'Symmetrizer_JIT',
           'Symmetrizer_Tensor', 'Symmetrizer_Tensor_Optimized', 'Symmetrizer_Triton']

"""
This class is used to symmetrize the basis functions in the A basis.
The basis functions are symmetrized by taking the product of the basis functions
"""

class Symmetrizer_JIT(nn.Module):
    """ This symmetrizer is implemented in JIT mode. """ 
    def __init__(self, max_nu: int, max_l: int, l_list: torch.Tensor):
        super().__init__()

        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        _, vg2, vi2, pf2, n2 = find_combo_vectors_nu2(max_l)
        self.register_buffer("vector_groups_2", vg2)
        self.register_buffer("vector_idx_2", vi2)
        self.register_buffer("prefactors_2", pf2)
        self.n2_start = 1
        self.n3_start = 1 + n2

        _, vg3, vi3, pf3, n3 = find_combo_vectors_nu3(max_l)
        self.register_buffer("vector_groups_3", vg3)
        self.register_buffer("vector_idx_3", vi3)
        self.register_buffer("prefactors_3", pf3)
        self.n4_start = 1 + n2 + n3

        _, vg4, vi4, pf4, n4 = find_combo_vectors_nu4(max_l)
        self.register_buffer("vector_groups_4", vg4)
        self.register_buffer("vector_idx_4", vi4)
        self.register_buffer("prefactors_4", pf4)

        # Initialize buffers for each nu value
        self.n_angular_sym = 1
        if max_nu >= 2:
            self.n_angular_sym += n2
        if max_nu >= 3:
            self.n_angular_sym += n3
        if max_nu == 4:
            self.n_angular_sym += n4

        # Register l_list as a buffer
        l_list_tensor = torch.tensor([l for l in l_list], dtype=torch.int64)
        self.register_buffer('l_list_tensor', l_list_tensor)

    @torch.jit.export
    def forward(self, node_attr: torch.Tensor) -> torch.Tensor:
        num_nodes, n_radial, _, n_chanel = node_attr.size()
        sym_node_attr = torch.zeros((num_nodes, n_radial, self.n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1
        if self.max_nu >= 2:
            for i, item in enumerate(self.vector_groups_2):
                prefactor = self.prefactors_2[i]
                idx = self.vector_idx_2[i]

                # Convert item to list of tuples
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n2_start, :] += prefactor * product
        if self.max_nu >= 3:
            for i, item in enumerate(self.vector_groups_3):
                prefactor = self.prefactors_3[i]
                idx = self.vector_idx_3[i]
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n3_start, :] += prefactor * product
        if self.max_nu == 4:
            for i, item in enumerate(self.vector_groups_4):
                prefactor = self.prefactors_4[i]
                idx = self.vector_idx_4[i]
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n4_start, :] += prefactor * product
        return sym_node_attr

    @torch.jit.export
    def _get_index_from_l_list(self, lxlylz: torch.Tensor) -> int:
        return torch.where((self.l_list_tensor == lxlylz).all(dim=1))[0][0].item()


class Symmetrizer(nn.Module):
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")
        self.vec_dict_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2]  = find_combo_vectors_nu2(self.max_l)[0]
        if max_nu >= 3:
            self.vec_dict_allnu[3]  = find_combo_vectors_nu3(self.max_l)[0]
        if max_nu == 4:
            self.vec_dict_allnu[4]  = find_combo_vectors_nu4(self.max_l)[0]

        self.indice_dict_allnu = None 
        self._get_indices_allnu()

    def _get_indices_allnu(self):
        self.indice_dict_allnu = {}
        for nu in range(2, self.max_nu + 1):
            self.indice_dict_allnu[nu] = {}
            for i, (l_key, lxlylz_list) in enumerate(self.vec_dict_allnu[nu].items()):
                self.indice_dict_allnu[nu][l_key] = []
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    # append to the dictionary
                    self.indice_dict_allnu[nu][l_key].append([indices, prefactor])

    def forward(self, node_attr: torch.Tensor):
        try:
            self.indice_dict_allnu
        except AttributeError:
           self._get_indices_allnu()

        num_nodes, n_radial, _, n_chanel = node_attr.size()
        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        for nu in range(2, self.max_nu + 1):
            for i, (_, indices_list) in enumerate(self.indice_dict_allnu[nu].items()):
                for item in indices_list:
                    indices, prefactor = item[0], item[-1]
                    product = torch.prod(node_attr[:, :, indices, :], dim=2)
                    # somehow MPS doesn't like torch.prod, as it uses cumprod during autograd.
                    # one can use the following:
                    #product = node_attr[:, :, indices[0], :]
                    #for idx in indices[1:]:
                    #    product = product * node_attr[:, :, idx, :]
                    sym_node_attr[:, :, i + n_sym_node_attr, :] += prefactor * product
            n_sym_node_attr += len(self.indice_dict_allnu[nu])

        return sym_node_attr

class Symmetrizer_Vectorized(nn.Module):
    """Vectorized Symmetrizer that batches combinations per order using tensor ops."""
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        l_list_tuples = [tuple(l) for l in l_list]
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")

        self.vec_dict_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2] = find_combo_vectors_nu2(self.max_l)[0]
        if max_nu >= 3:
            self.vec_dict_allnu[3] = find_combo_vectors_nu3(self.max_l)[0]
        if max_nu == 4:
            self.vec_dict_allnu[4] = find_combo_vectors_nu4(self.max_l)[0]

        # Keep this as a Python int so TorchDynamo doesn't wrap it into a fake scalar.
        total_slots = sum(len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1))
        self.n_angular_sym = 1 + int(total_slots)
        self.order_offsets = {}
        self.n_slots_per_order = {}
        self._build_blocks()

    def _build_blocks(self):
        """Precompute tensors for vectorized contractions per nu."""
        offset = 1
        for nu in range(2, self.max_nu + 1):
            vec_dict = self.vec_dict_allnu.get(nu)
            if vec_dict is None:
                continue

            combo_indices = []
            combo_prefactors = []
            combo_slots = []

            for slot_idx, (_, combo_list) in enumerate(vec_dict.items()):
                for item in combo_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    combo_indices.append(indices)
                    combo_prefactors.append(prefactor)
                    combo_slots.append(slot_idx)

            if not combo_indices:
                continue

            indices_tensor = torch.tensor(combo_indices, dtype=torch.long)
            prefactors_tensor = torch.tensor(combo_prefactors, dtype=torch.get_default_dtype())
            slots_tensor = torch.tensor(combo_slots, dtype=torch.long)
            counts = torch.bincount(
                slots_tensor,
                minlength=len(vec_dict),
            )
            slot_offsets = torch.zeros(len(vec_dict) + 1, dtype=torch.long)
            slot_offsets[1:] = torch.cumsum(counts, dim=0)

            self.register_buffer(f"indices_nu{nu}", indices_tensor)
            self.register_buffer(f"prefactors_nu{nu}", prefactors_tensor)
            self.register_buffer(f"slots_nu{nu}", slots_tensor)
            self.register_buffer(f"slot_offsets_nu{nu}", slot_offsets)
            self.order_offsets[nu] = offset
            self.n_slots_per_order[nu] = len(vec_dict)
            offset += len(vec_dict)

    def forward(self, node_attr: torch.Tensor):
        num_nodes, n_radial, _, n_channel = node_attr.size()

        sym_node_attr = node_attr.new_zeros((num_nodes, n_radial, self.n_angular_sym, n_channel))
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]

        for nu in range(2, self.max_nu + 1):
            indices = getattr(self, f"indices_nu{nu}", None)
            prefactors = getattr(self, f"prefactors_nu{nu}", None)
            slots = getattr(self, f"slots_nu{nu}", None)
            n_slots = self.n_slots_per_order.get(nu)
            if indices is None or prefactors is None or slots is None or n_slots is None:
                continue

            # indices = indices.to(node_attr.device)
            # prefactors = prefactors.to(node_attr.device, dtype=node_attr.dtype)
            # slots = slots.to(node_attr.device)

            gathered = node_attr[:, :, indices, :]
            products = torch.prod(gathered, dim=3)
            weighted = products * prefactors.view(1, 1, -1, 1)

            slice_out = node_attr.new_zeros((num_nodes, n_radial, n_slots, n_channel))
            scatter_idx = slots.view(1, 1, -1, 1).expand(num_nodes, n_radial, -1, n_channel)
            slice_out.scatter_add_(2, scatter_idx, weighted)

            offset = self.order_offsets[nu]
            sym_node_attr[:, :, offset:offset + n_slots, :] = slice_out

        return sym_node_attr

class Symmetrizer_Tensor(nn.Module):
    """ This symmetrizer is implemented using tensor operations. 
        Not performant for nu=4, but should be fine for smaller nu and large max_l.
    """
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l
        self.l_list = l_list
        self.n_l = len(l_list)

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")
        self.vec_dict_allnu = {}
        self.sym_tensor_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2]  = find_combo_vectors_nu2(self.max_l)[0]
            # 3D tensor of shape (n_l, n_l, len(vec_dict_allnu[2]))
            self.sym_tensor_allnu[2] = torch.zeros((self.n_l, self.n_l, len(self.vec_dict_allnu[2])))
            # loop through the dictionary and assign the values to the tensor 
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[2].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[2][indices[0], indices[1], i] = prefactor 

        if max_nu >= 3:
            self.vec_dict_allnu[3]  = find_combo_vectors_nu3(self.max_l)[0]

            self.sym_tensor_allnu[3] = torch.zeros((self.n_l, self.n_l, self.n_l, len(self.vec_dict_allnu[3])))
            # loop through the dictionary and assign the values to the tensor
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[3].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[3][indices[0], indices[1], indices[2], i] = prefactor

        if max_nu == 4:
            self.vec_dict_allnu[4]  = find_combo_vectors_nu4(self.max_l)[0]

            self.sym_tensor_allnu[4] = torch.zeros((self.n_l, self.n_l, self.n_l, self.n_l, len(self.vec_dict_allnu[4])))
            # loop through the dictionary and assign the values to the tensor
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[4].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[4][indices[0], indices[1], indices[2], indices[3], i] = prefactor

    def _apply(self, fn):
        """Override _apply to move tensors in sym_tensor_allnu dictionary to device."""
        # Call parent _apply first to handle registered buffers/parameters
        super()._apply(fn)
        
        # Move tensors in sym_tensor_allnu dictionary
        for key in self.sym_tensor_allnu:
            if isinstance(self.sym_tensor_allnu[key], torch.Tensor):
                self.sym_tensor_allnu[key] = fn(self.sym_tensor_allnu[key])

        for key in self.vec_dict_allnu:
            if isinstance(self.vec_dict_allnu[key], torch.Tensor):
                self.vec_dict_allnu[key] = fn(self.vec_dict_allnu[key])
        
        return self

    def forward(self, node_attr: torch.Tensor):
        num_nodes, n_radial, n_l, n_chanel = node_attr.size()
        assert n_l == self.n_l

        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        if self.max_nu >= 2:
            # for nu==2, node_attr_2 is the product of node_attr
            # node_attr: [num_nodes, n_radial, n_l, n_l, n_channel]
            node_attr_2 = torch.einsum('ijlk,ijmk->ijlmk', node_attr, node_attr) 
            sym_tensor = self.sym_tensor_allnu[2]
            n_sym_node_attr_now = sym_tensor.shape[2]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmk,lma->ijak', node_attr_2, sym_tensor) 
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 3:
            node_attr_3 = torch.einsum('ijlmk,ijnk->ijlmnk', node_attr_2, node_attr)
            sym_tensor = self.sym_tensor_allnu[3]
            n_sym_node_attr_now = sym_tensor.shape[3]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmnk,lmna->ijak', node_attr_3, sym_tensor)
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 4:
            node_attr_4 = torch.einsum('ijlmnk,ijok->ijlmnok', node_attr_3, node_attr)
            sym_tensor = self.sym_tensor_allnu[4]
            n_sym_node_attr_now = sym_tensor.shape[4]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmnok,lmnoa->ijak', node_attr_4, sym_tensor)
            n_sym_node_attr += n_sym_node_attr_now

        return sym_node_attr


class Symmetrizer_Tensor_Optimized(Symmetrizer_Tensor):
    """ This symmetrizer is implemented using optimized tensor operations with opt_einsum.
        Uses fused einsum operations to avoid storing large intermediate tensors.
    """
    def forward(self, node_attr: torch.Tensor):
        num_nodes, n_radial, n_l, n_chanel = node_attr.size()
        assert n_l == self.n_l

        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        # with shared_intermediates():
        if self.max_nu >= 2:
            # Fuse the two einsum operations to avoid storing node_attr_2
            sym_tensor = self.sym_tensor_allnu[2]
            n_sym_node_attr_now = sym_tensor.shape[2]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum(
                'ijlk,ijmk,lma->ijak', node_attr, node_attr, sym_tensor
            )
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 3:
            # Fuse all three einsum operations to avoid storing node_attr_2 and node_attr_3
            sym_tensor = self.sym_tensor_allnu[3]
            n_sym_node_attr_now = sym_tensor.shape[3]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum(
                'ijlk,ijmk,ijnk,lmna->ijak', node_attr, node_attr, node_attr, sym_tensor
            )
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 4:
            # Fuse all four einsum operations to avoid storing node_attr_2, node_attr_3, and node_attr_4
            sym_tensor = self.sym_tensor_allnu[4]
            n_sym_node_attr_now = sym_tensor.shape[4]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum(
                'ijlk,ijmk,ijnk,ijok,lmnoa->ijak', node_attr, node_attr, node_attr, node_attr, sym_tensor
            )
            n_sym_node_attr += n_sym_node_attr_now

        return sym_node_attr


class Symmetrizer_Triton(Symmetrizer_Vectorized):
    """Symmetrizer implementation that accelerates the vectorized formulation with Triton."""

    def __init__(
        self,
        max_nu: int,
        max_l: int,
        l_list: list,
        block_rows: int = 128,
        block_combos: int = 32,
    ):
        super().__init__(max_nu=max_nu, max_l=max_l, l_list=l_list)
        self.block_rows = block_rows
        self.block_combos = block_combos
        self.use_triton = HAS_TRITON
        if not HAS_TRITON:
            warnings.warn(
                "Triton is not available; Symmetrizer_Triton will fall back to the "
                "vectorized implementation.",
                RuntimeWarning,
            )

        self.slot_iters_per_order = {}

        # Convert index/slot buffers to int32 to avoid casting on every launch
        for nu in range(2, self.max_nu + 1):
            idx_name = f"indices_nu{nu}"
            slot_name = f"slots_nu{nu}"
            offset_name = f"slot_offsets_nu{nu}"
            if hasattr(self, idx_name):
                tensor = getattr(self, idx_name)
                if tensor is not None:
                    delattr(self, idx_name)
                    self.register_buffer(idx_name, tensor.to(torch.int32))
            if hasattr(self, slot_name):
                tensor = getattr(self, slot_name)
                if tensor is not None:
                    delattr(self, slot_name)
                    self.register_buffer(slot_name, tensor.to(torch.int32))
            if hasattr(self, offset_name):
                tensor = getattr(self, offset_name)
                if tensor is not None:
                    delattr(self, offset_name)
                    self.register_buffer(offset_name, tensor.to(torch.int32))
                    counts = tensor[1:] - tensor[:-1]
                    max_count = counts.max().item() if counts.numel() > 0 else 0
                    self.slot_iters_per_order[nu] = max(
                        1, math.ceil(max_count / self.block_combos)
                    )
                else:
                    self.slot_iters_per_order[nu] = 1

    def forward(self, node_attr: torch.Tensor):
        if (
            not self.use_triton
            or node_attr.device.type != "cuda"
            or node_attr.numel() == 0
        ):
            return super().forward(node_attr)
        return self._forward_triton(node_attr)

    def _forward_triton(self, node_attr: torch.Tensor) -> torch.Tensor:
        num_nodes, n_radial, n_l, n_channel = node_attr.size()
        node_attr = node_attr.contiguous()
        node_flat = node_attr.permute(0, 1, 3, 2).reshape(-1, n_l).contiguous()
        out_flat = torch.zeros(
            (node_flat.shape[0], self.n_angular_sym),
            dtype=node_attr.dtype,
            device=node_attr.device,
        )
        out_flat[:, 0] = node_attr[:, :, 0, :].reshape(-1)

        for nu in range(2, self.max_nu + 1):
            indices = getattr(self, f"indices_nu{nu}", None)
            prefactors = getattr(self, f"prefactors_nu{nu}", None)
            slot_offsets = getattr(self, f"slot_offsets_nu{nu}", None)
            if (
                indices is None
                or prefactors is None
                or self.n_slots_per_order.get(nu) is None
                or slot_offsets is None
            ):
                continue

            pref = prefactors
            if pref.dtype != node_attr.dtype:
                pref = pref.to(node_attr.dtype)
            run_symmetrizer_order_kernel(
                node_flat,
                out_flat,
                indices,
                pref,
                slot_offsets,
                self.order_offsets[nu],
                block_rows=self.block_rows,
                block_combos=self.block_combos,
                max_iters=self.slot_iters_per_order.get(nu, 1),
            )

        sym_node_attr = (
            out_flat.view(num_nodes, n_radial, n_channel, self.n_angular_sym)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        return sym_node_attr
