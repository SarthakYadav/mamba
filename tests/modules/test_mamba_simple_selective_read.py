import pytest
import torch

from mamba_ssm.modules import mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba


def _require_fast_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if mamba_simple.causal_conv1d_fn is None:
        pytest.skip("causal-conv1d is required for the fused Mamba fast path")


def test_mamba_selective_read_false_uses_static_c_projection():
    _require_fast_path()
    torch.manual_seed(0)

    model = Mamba(
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=4,
        selective_read=False,
        use_fast_path=True,
        device="cuda",
        dtype=torch.float16,
    )

    assert model.x_proj.out_features == model.dt_rank + model.d_state
    assert model.C.shape == (model.d_inner, model.d_state)
    assert model.C.dtype == torch.float32


def test_mamba_selective_read_false_fused_half_forward_backward():
    _require_fast_path()
    torch.manual_seed(1)

    batch, seqlen, d_model = 2, 32, 64
    model = Mamba(
        d_model=d_model,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=4,
        selective_read=False,
        use_fast_path=True,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn(batch, seqlen, d_model, device="cuda", dtype=torch.float16, requires_grad=True)

    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16

    y.float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.C.grad is not None
    assert torch.isfinite(model.C.grad).all()
