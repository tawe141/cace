import torch
import pytest

from cace.modules.radial_transform import (
    SharedRadialLinearTransform,
    SharedRadialLinearTransformV2,
)


@pytest.mark.parametrize("channel_dim", [None, 3])
def test_shared_radial_linear_transform_v2_matches_v1(channel_dim):
    torch.manual_seed(1234)

    max_l = 4
    radial_dim = 5
    radial_embedding_dim = 6
    embedding_dim = channel_dim if channel_dim is not None else 7
    n_nodes = 3

    baseline = SharedRadialLinearTransform(max_l, radial_dim, radial_embedding_dim, channel_dim)
    optimized = SharedRadialLinearTransformV2(max_l, radial_dim, radial_embedding_dim, channel_dim)
    optimized.load_state_dict(baseline.state_dict())

    angular_dim = int(baseline.angular_dim_groups[-1, 1].item())
    x = torch.randn(n_nodes, radial_dim, angular_dim, embedding_dim, dtype=torch.float32)

    baseline_output = baseline(x)
    optimized_output = optimized(x)

    torch.testing.assert_close(baseline_output, optimized_output, rtol=1e-5, atol=1e-6)
