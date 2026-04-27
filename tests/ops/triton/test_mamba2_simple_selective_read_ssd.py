import pytest
import torch

from mamba_ssm.ops.triton import ssd_combined
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_split_conv1d_scan_combined,
    mamba_split_conv1d_scan_ref,
)


def _require_fused_wrapper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if ssd_combined.causal_conv1d_fn is None:
        pytest.skip("causal-conv1d is required for the reference wrapper")
    if ssd_combined.causal_conv1d_fwd_function is None or ssd_combined.causal_conv1d_bwd_function is None:
        pytest.skip("causal-conv1d fused kernels are required for the combined wrapper")


def _clone_inputs(*args):
    return tuple(arg.detach().clone().requires_grad_(True) for arg in args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ngroups", [1, 2])
def test_mamba2_simple_static_c_wrapper_parity(dtype, ngroups):
    _require_fused_wrapper()
    torch.manual_seed(0)

    device = "cuda"
    batch = 2
    seqlen = 32
    nheads = 4
    headdim = 32
    dim = nheads * headdim
    dstate = 8
    dconv = 4
    chunk_size = 16
    atol = rtol = 1e-2

    zxbcdt = (torch.randn(batch, seqlen, 2 * dim + ngroups * dstate + nheads, device=device, dtype=dtype) / 5).requires_grad_()
    conv1d_weight = (torch.randn(dim + ngroups * dstate, dconv, device=device, dtype=dtype) / 5).requires_grad_()
    conv1d_bias = (torch.randn(dim + ngroups * dstate, device=device, dtype=dtype) / 5).requires_grad_()
    dt_bias = (torch.randn(nheads, device=device, dtype=dtype) / 5).requires_grad_()
    A = (-torch.exp(torch.randn(nheads, device=device, dtype=torch.float32) / 5)).requires_grad_()
    D = (torch.randn(nheads, device=device, dtype=torch.float32) / 5).requires_grad_()
    static_C = (torch.randn(ngroups, dstate, device=device, dtype=dtype) / 5).requires_grad_()

    zxbcdt_ref, conv1d_weight_ref, conv1d_bias_ref, dt_bias_ref, A_ref, D_ref, static_C_ref = _clone_inputs(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, static_C
    )

    out = mamba_split_conv1d_scan_combined(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size=chunk_size,
        activation="silu",
        headdim=headdim,
        ngroups=ngroups,
        static_C=static_C,
    )
    out_ref = mamba_split_conv1d_scan_ref(
        zxbcdt_ref,
        conv1d_weight_ref,
        conv1d_bias_ref,
        dt_bias_ref,
        A_ref,
        D_ref,
        chunk_size=chunk_size,
        activation="silu",
        headdim=headdim,
        ngroups=ngroups,
        static_C=static_C_ref,
    )

    assert out.shape == out_ref.shape
    assert torch.allclose(out.float(), out_ref.float(), atol=atol, rtol=rtol)

    out.float().square().mean().backward()
    out_ref.float().square().mean().backward()

    grads = [
        ("zxbcdt", zxbcdt.grad, zxbcdt_ref.grad),
        ("conv1d_weight", conv1d_weight.grad, conv1d_weight_ref.grad),
        ("conv1d_bias", conv1d_bias.grad, conv1d_bias_ref.grad),
        ("dt_bias", dt_bias.grad, dt_bias_ref.grad),
        ("A", A.grad, A_ref.grad),
        ("D", D.grad, D_ref.grad),
        ("static_C", static_C.grad, static_C_ref.grad),
    ]
    for name, grad, grad_ref in grads:
        assert grad is not None, f"missing grad for {name}"
        assert grad_ref is not None, f"missing reference grad for {name}"
        assert torch.isfinite(grad).all(), f"non-finite grad for {name}"
        assert torch.isfinite(grad_ref).all(), f"non-finite reference grad for {name}"
        assert torch.allclose(grad.float(), grad_ref.float(), atol=atol, rtol=rtol), f"{name} grad mismatch"
