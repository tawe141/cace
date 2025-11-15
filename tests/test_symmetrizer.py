"""
Test suite for Symmetrizer classes to verify backward compatibility
and numerical equivalence between implementations.
"""

import pytest
import torch
import numpy as np
from cace.modules import (
    Symmetrizer,
    Symmetrizer_Tensor,
    Symmetrizer_Tensor_Optimized,
    AngularComponent
)


@pytest.fixture
def device():
    """Fixture to determine available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dtype():
    """Fixture for default dtype."""
    return torch.float32


def get_l_list(max_l):
    """Helper function to get l_list from AngularComponent."""
    angular = AngularComponent(max_l)
    return angular.get_lxlylz_list()


# @pytest.mark.parametrize("max_l", [1, 2, 3, 4])
# @pytest.mark.parametrize("max_nu", [2, 3, 4])
# @pytest.mark.parametrize("num_nodes", [3, 10, 50])
# @pytest.mark.parametrize("n_radial", [4, 8])
# @pytest.mark.parametrize("n_channel", [16, 32])
# def test_symmetrizer_tensor_optimized_vs_tensor(
#     max_l, max_nu, num_nodes, n_radial, n_channel, device, dtype
# ):
#     """Test that Symmetrizer_Tensor_Optimized produces same results as Symmetrizer_Tensor."""
#     if max_nu > max_l + 1:
#         pytest.skip(f"max_nu={max_nu} > max_l+1={max_l+1} is not a valid combination")
    
#     l_list = get_l_list(max_l)
#     n_l = len(l_list)
    
#     # Create symmetrizer instances
#     symmetrizer_tensor = Symmetrizer_Tensor(max_nu, max_l, l_list)
#     symmetrizer_optimized = Symmetrizer_Tensor_Optimized(max_nu, max_l, l_list)
    
#     # Move to device
#     symmetrizer_tensor = symmetrizer_tensor.to(device)
#     symmetrizer_optimized = symmetrizer_optimized.to(device)
    
#     # Create random input tensor
#     torch.manual_seed(42)
#     node_attr = torch.randn(num_nodes, n_radial, n_l, n_channel, dtype=dtype, device=device)
    
#     # Compute outputs
#     with torch.no_grad():
#         output_tensor = symmetrizer_tensor(node_attr)
#         output_optimized = symmetrizer_optimized(node_attr)
    
#     # Check shapes match
#     assert output_tensor.shape == output_optimized.shape, \
#         f"Shape mismatch: {output_tensor.shape} vs {output_optimized.shape}"
    
#     # Check numerical equivalence (may have small differences due to different contraction order)
#     # Use relatively relaxed tolerance due to floating point accumulation differences
#     # For larger max_l and max_nu, differences can be slightly larger due to more complex contractions
#     # and larger tensor sizes leading to more floating point operations
#     if max_nu >= 4 or (max_nu >= 3 and max_l >= 4):
#         atol = 1e-4
#         rtol = 1e-4
#     elif max_nu >= 3:
#         atol = 1e-5
#         rtol = 1e-5
#     else:
#         atol = 1e-6
#         rtol = 1e-5
#     assert torch.allclose(output_tensor, output_optimized, rtol=rtol, atol=atol), \
#         f"Output mismatch for max_l={max_l}, max_nu={max_nu}, shape={node_attr.shape}, max_diff={(output_tensor - output_optimized).abs().max().item():.2e}"


@pytest.mark.parametrize("max_l", [1, 2, 3, 4])
@pytest.mark.parametrize("max_nu", [2, 3, 4])
@pytest.mark.parametrize("num_nodes", [3, 10])
@pytest.mark.parametrize("n_radial", [4, 8])
@pytest.mark.parametrize("n_channel", [16, 32])
def test_symmetrizer_tensor_optimized_vs_symmetrizer(
    max_l, max_nu, num_nodes, n_radial, n_channel, device, dtype
):
    """Test that Symmetrizer_Tensor_Optimized produces same results as Symmetrizer."""
    # if max_nu > max_l + 1:
    #     pytest.skip(f"max_nu={max_nu} > max_l+1={max_l+1} is not a valid combination")
    
    l_list = get_l_list(max_l)
    n_l = len(l_list)
    
    # Create symmetrizer instances
    symmetrizer = Symmetrizer(max_nu, max_l, l_list)
    symmetrizer_optimized = Symmetrizer_Tensor_Optimized(max_nu, max_l, l_list)
    
    # Move to device
    symmetrizer = symmetrizer.to(device)
    symmetrizer_optimized = symmetrizer_optimized.to(device)
    
    # Create random input tensor
    torch.manual_seed(42)
    node_attr = torch.randn(num_nodes, n_radial, n_l, n_channel, dtype=dtype, device=device)
    
    # Compute outputs
    with torch.no_grad():
        output_symmetrizer = symmetrizer(node_attr)
        output_optimized = symmetrizer_optimized(node_attr)
    
    # Check shapes match
    assert output_symmetrizer.shape == output_optimized.shape, \
        f"Shape mismatch: {output_symmetrizer.shape} vs {output_optimized.shape}"
    
    # Check numerical equivalence
    # Use relatively relaxed tolerance due to different computation paths
    assert torch.allclose(output_symmetrizer, output_optimized, rtol=1e-4, atol=1e-5), \
        f"Output mismatch for max_l={max_l}, max_nu={max_nu}, shape={node_attr.shape}"


@pytest.mark.parametrize("max_l", [1, 2, 3, 4])
@pytest.mark.parametrize("max_nu", [2, 3, 4])
def test_symmetrizer_tensor_optimized_device_movement(max_l, max_nu, dtype):
    """Test that device movement works correctly for Symmetrizer_Tensor_Optimized."""
    # if max_nu > max_l + 1:
    #     pytest.skip(f"max_nu={max_nu} > max_l+1={max_l+1} is not a valid combination")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    l_list = get_l_list(max_l)
    n_l = len(l_list)
    
    # Create on CPU
    symmetrizer = Symmetrizer_Tensor_Optimized(max_nu, max_l, l_list)
    
    # Move to CUDA
    symmetrizer = symmetrizer.cuda()
    
    # Create input on CUDA
    torch.manual_seed(42)
    node_attr = torch.randn(5, 8, n_l, 16, dtype=dtype, device='cuda')
    
    # Should work without errors
    with torch.no_grad():
        output = symmetrizer(node_attr)
    
    assert output.device.type == 'cuda', "Output should be on CUDA device"


@pytest.mark.parametrize("max_l", [1, 2, 3, 4])
@pytest.mark.parametrize("max_nu", [2, 3, 4])
def test_symmetrizer_tensor_optimized_gradient(max_l, max_nu, device, dtype):
    """Test that gradients can be computed correctly."""
    # if max_nu > max_l + 1:
    #     pytest.skip(f"max_nu={max_nu} > max_l+1={max_l+1} is not a valid combination")
    
    l_list = get_l_list(max_l)
    n_l = len(l_list)
    
    # Create symmetrizer
    symmetrizer = Symmetrizer_Tensor_Optimized(max_nu, max_l, l_list).to(device)
    
    # Create input with requires_grad
    torch.manual_seed(42)
    node_attr = torch.randn(5, 8, n_l, 16, dtype=dtype, device=device, requires_grad=True)
    
    # Forward pass
    output = symmetrizer(node_attr)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert node_attr.grad is not None, "Gradients should be computed"
    assert not torch.isnan(node_attr.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(node_attr.grad).any(), "Gradients should not contain Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

