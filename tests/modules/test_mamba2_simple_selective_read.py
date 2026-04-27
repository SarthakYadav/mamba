import pytest
import torch

from mamba_ssm.modules import mamba2_simple as mamba2_simple_module
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.ops.triton import ssd_combined


def _require_mamba2_simple():
    if mamba2_simple_module.RMSNormGated is None:
        pytest.skip("Triton RMSNormGated is required for Mamba2Simple")


def _require_cuda():
    _require_mamba2_simple()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _require_fused_path():
    _require_cuda()
    if ssd_combined.causal_conv1d_fwd_function is None or ssd_combined.causal_conv1d_bwd_function is None:
        pytest.skip("causal-conv1d is required for the fused Mamba2Simple path")


def test_mamba2_simple_selective_read_false_shapes():
    _require_mamba2_simple()

    model = Mamba2Simple(
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=2,
        selective_read=False,
        use_mem_eff_path=False,
    )

    assert model.selective_read is False
    assert model.C.shape == (model.ngroups, model.d_state)
    assert model.in_proj.out_features == 2 * model.d_inner + model.ngroups * model.d_state + model.nheads
    assert model.conv1d.weight.shape[0] == model.d_inner + model.ngroups * model.d_state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mamba2_simple_selective_read_false_nonfused_forward_backward(dtype):
    _require_cuda()
    torch.manual_seed(0)

    batch, seqlen, d_model = 2, 32, 64
    model = Mamba2Simple(
        d_model=d_model,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=2,
        selective_read=False,
        use_mem_eff_path=False,
        device="cuda",
        dtype=dtype,
    )
    x = torch.randn(batch, seqlen, d_model, device="cuda", dtype=dtype, requires_grad=True)

    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == dtype

    y.float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.C.grad is not None
    assert torch.isfinite(model.C.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mamba2_simple_selective_read_false_fused_forward_backward(dtype):
    _require_fused_path()
    torch.manual_seed(1)

    batch, seqlen, d_model = 2, 32, 64
    model = Mamba2Simple(
        d_model=d_model,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=2,
        selective_read=False,
        use_mem_eff_path=True,
        device="cuda",
        dtype=dtype,
    )
    x = torch.randn(batch, seqlen, d_model, device="cuda", dtype=dtype, requires_grad=True)

    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == dtype

    y.float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.C.grad is not None
    assert torch.isfinite(model.C.grad).all()
