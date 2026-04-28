# Copyright (c) 2026 Sarthak Yadav

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from einops import rearrange, repeat

try:
    from triton.testing import do_bench
except ImportError:
    do_bench = None

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.ops.triton.ssd_combined import _mamba_split_conv1d_scan_static_c_legacy


FAMILY_DISPLAY = {
    "mamba": "Mamba",
    "mamba2_simple": "Mamba2Simple",
}

DTYPE_CHOICES = ["float16", "bfloat16", "float32"]


@dataclass
class BenchmarkResult:
    family: str
    selective_read: bool
    pass_name: str
    dtype: str
    batch: int
    seqlen: int
    d_model: int
    ms: float | None
    toks_per_sec: float | None
    peak_memory_mb: float | None = None
    benchmark_group: str = "matrix"
    static_c_impl: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the 2x2 selective-read matrix for Mamba and Mamba2Simple "
            "across batch sizes, sequence lengths, embedding dimensions, and dtypes."
        )
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048, 4096])
    parser.add_argument("--d-models", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument(
        "--dtypes",
        choices=DTYPE_CHOICES,
        nargs="+",
        default=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--passes",
        choices=["forward", "forward_backward"],
        nargs="+",
        default=["forward", "forward_backward"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--ngroups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument(
        "--families",
        choices=sorted(FAMILY_DISPLAY.keys()),
        nargs="+",
        default=["mamba", "mamba2_simple"],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately on the first benchmark failure instead of recording it.",
    )
    return parser.parse_args()


def benchmark_ms(fn: Callable[[], None], *, warmup: int, rep: int) -> float:
    if do_bench is not None:
        return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return float(timings[len(timings) // 2])


def make_model(
    family: str,
    *,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int,
    selective_read: bool,
    device: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if family == "mamba":
        return Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            selective_read=selective_read,
            use_fast_path=True,
            device=device,
            dtype=dtype,
        )
    if family == "mamba2_simple":
        return Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            selective_read=selective_read,
            use_mem_eff_path=True,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported family: {family}")


def make_input(*, batch: int, seqlen: int, d_model: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(batch, seqlen, d_model, device=device, dtype=dtype)


def _reset_cuda_state() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _time_with_memory(fn: Callable[[], None], *, warmup: int, rep: int) -> tuple[float, float]:
    """Run the bench while tracking peak device memory (in MiB)."""
    _reset_cuda_state()
    ms = benchmark_ms(fn, warmup=warmup, rep=rep)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ms, peak_mb


def run_case(
    *,
    family: str,
    selective_read: bool,
    benchmark_group: str,
    pass_name: str,
    dtype_str: str,
    batch: int,
    seqlen: int,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> BenchmarkResult:
    model = make_model(
        family,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        selective_read=selective_read,
        device=device,
        dtype=dtype,
    )
    x_data = make_input(batch=batch, seqlen=seqlen, d_model=d_model, device=device, dtype=dtype)

    if pass_name == "forward":
        model.eval()

        @torch.inference_mode()
        def fn() -> None:
            model(x_data)
    elif pass_name == "forward_backward":
        model.train()

        def fn() -> None:
            model.zero_grad(set_to_none=True)
            x = x_data.detach().clone().requires_grad_(True)
            y = model(x)
            y.float().square().mean().backward()
    else:
        raise ValueError(f"Unsupported pass: {pass_name}")

    ms, peak_mb = _time_with_memory(fn, warmup=warmup, rep=rep)
    toks_per_sec = batch * seqlen / (ms / 1000.0)
    return BenchmarkResult(
        family=family,
        selective_read=selective_read,
        pass_name=pass_name,
        dtype=dtype_str,
        batch=batch,
        seqlen=seqlen,
        d_model=d_model,
        ms=ms,
        toks_per_sec=toks_per_sec,
        peak_memory_mb=peak_mb,
        benchmark_group=benchmark_group,
    )


def _run_legacy_static_c_mamba2_simple(model: Mamba2Simple, x: torch.Tensor) -> torch.Tensor:
    zxbcdt = model.in_proj(x)
    A = -torch.exp(model.A_log)
    initial_states = (
        repeat(model.init_states, "... -> b ...", b=x.shape[0]) if model.learnable_init_states else None
    )
    return _mamba_split_conv1d_scan_static_c_legacy(
        zxbcdt,
        rearrange(model.conv1d.weight, "d 1 w -> d w"),
        model.conv1d.bias,
        model.dt_bias,
        A,
        model.D,
        chunk_size=model.chunk_size,
        initial_states=initial_states,
        seq_idx=None,
        dt_limit=model.dt_limit,
        return_final_states=False,
        activation=model.activation,
        rmsnorm_weight=model.norm.weight,
        rmsnorm_eps=model.norm.eps,
        outproj_weight=model.out_proj.weight,
        outproj_bias=model.out_proj.bias,
        headdim=model.headdim,
        ngroups=model.ngroups,
        norm_before_gate=False,
        static_C=model.C,
    )


def run_static_c_baseline_case(
    *,
    impl: str,
    pass_name: str,
    dtype_str: str,
    batch: int,
    seqlen: int,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> BenchmarkResult:
    model = make_model(
        "mamba2_simple",
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        selective_read=False,
        device=device,
        dtype=dtype,
    )
    x_data = make_input(batch=batch, seqlen=seqlen, d_model=d_model, device=device, dtype=dtype)

    if impl == "reduced":
        if pass_name == "forward":
            model.eval()

            @torch.inference_mode()
            def fn() -> None:
                model(x_data)
        elif pass_name == "forward_backward":
            model.train()

            def fn() -> None:
                model.zero_grad(set_to_none=True)
                x = x_data.detach().clone().requires_grad_(True)
                y = model(x)
                y.float().square().mean().backward()
        else:
            raise ValueError(f"Unsupported pass: {pass_name}")
    elif impl == "legacy_reference":
        if pass_name == "forward":
            model.eval()

            @torch.inference_mode()
            def fn() -> None:
                _run_legacy_static_c_mamba2_simple(model, x_data)
        elif pass_name == "forward_backward":
            model.train()

            def fn() -> None:
                model.zero_grad(set_to_none=True)
                x = x_data.detach().clone().requires_grad_(True)
                y = _run_legacy_static_c_mamba2_simple(model, x)
                y.float().square().mean().backward()
        else:
            raise ValueError(f"Unsupported pass: {pass_name}")
    else:
        raise ValueError(f"Unsupported impl: {impl}")

    ms, peak_mb = _time_with_memory(fn, warmup=warmup, rep=rep)
    toks_per_sec = batch * seqlen / (ms / 1000.0)
    return BenchmarkResult(
        family="mamba2_simple",
        selective_read=False,
        pass_name=pass_name,
        dtype=dtype_str,
        batch=batch,
        seqlen=seqlen,
        d_model=d_model,
        ms=ms,
        toks_per_sec=toks_per_sec,
        peak_memory_mb=peak_mb,
        benchmark_group="static_c_baseline",
        static_c_impl=impl,
    )


def format_metric(value: float | None) -> str:
    return f"{value:8.3f}" if value is not None else "   error "


def format_speedup(current: BenchmarkResult | None, baseline: BenchmarkResult | None) -> str:
    if current is None or baseline is None or current.ms is None or baseline.ms is None:
        return "   n/a  "
    return f"{baseline.ms / current.ms:8.3f}x"


def print_shape_block(
    results: list[BenchmarkResult],
    *,
    families: list[str],
    pass_name: str,
    dtype_str: str,
    batch: int,
    seqlen: int,
    d_model: int,
) -> None:
    print()
    print(f"[{pass_name}] dtype={dtype_str} batch={batch} seqlen={seqlen} d_model={d_model}")
    print(
        f"{'family':<14} {'read=True ms':>12} {'read=False ms':>13} "
        f"{'false/true':>11} {'mem_T MB':>10} {'mem_F MB':>10} "
        f"{'read=True tok/s':>16} {'read=False tok/s':>17}"
    )
    for family in families:
        family_results = {r.selective_read: r for r in results if r.family == family}
        read_true = family_results.get(True)
        read_false = family_results.get(False)
        read_true_toks = read_true.toks_per_sec if read_true is not None else None
        read_false_toks = read_false.toks_per_sec if read_false is not None else None
        read_true_mem = read_true.peak_memory_mb if read_true is not None else None
        read_false_mem = read_false.peak_memory_mb if read_false is not None else None
        print(
            f"{FAMILY_DISPLAY[family]:<14} "
            f"{format_metric(read_true.ms if read_true is not None else None):>12} "
            f"{format_metric(read_false.ms if read_false is not None else None):>13} "
            f"{format_speedup(read_false, read_true):>11} "
            f"{format_metric(read_true_mem):>10} "
            f"{format_metric(read_false_mem):>10} "
            f"{format_metric(read_true_toks):>16} "
            f"{format_metric(read_false_toks):>17}"
        )
        for result in (read_true, read_false):
            if result is not None and result.error is not None:
                print(f"  note: {FAMILY_DISPLAY[family]} selective_read={result.selective_read}: {result.error}")

    if set(families) >= {"mamba", "mamba2_simple"}:
        by_family_and_read = {(r.family, r.selective_read): r for r in results}
        for selective_read in [True, False]:
            mamba = by_family_and_read.get(("mamba", selective_read))
            mamba2 = by_family_and_read.get(("mamba2_simple", selective_read))
            if (
                mamba is not None
                and mamba2 is not None
                and mamba.ms is not None
                and mamba2.ms is not None
            ):
                print(
                    f"  cross-family selective_read={selective_read}: "
                    f"Mamba2Simple vs Mamba speedup = {mamba.ms / mamba2.ms:0.3f}x"
                )


def print_static_c_impl_block(
    results: list[BenchmarkResult],
    *,
    pass_name: str,
    dtype_str: str,
    batch: int,
    seqlen: int,
    d_model: int,
) -> None:
    if not results:
        return
    print()
    print(
        f"[{pass_name}] static-C baseline dtype={dtype_str} batch={batch} "
        f"seqlen={seqlen} d_model={d_model}"
    )
    print(f"{'impl':<18} {'ms':>12} {'mem MB':>10} {'tok/s':>16} {'reduced/legacy':>18}")
    by_impl = {r.static_c_impl: r for r in results}
    legacy = by_impl.get("legacy_reference")
    reduced = by_impl.get("reduced")
    for impl in ["legacy_reference", "reduced"]:
        result = by_impl.get(impl)
        print(
            f"{impl:<18} "
            f"{format_metric(result.ms if result is not None else None):>12} "
            f"{format_metric(result.peak_memory_mb if result is not None else None):>10} "
            f"{format_metric(result.toks_per_sec if result is not None else None):>16} "
            f"{format_speedup(reduced, legacy) if impl == 'reduced' else '':>18}"
        )
        if result is not None and result.error is not None:
            print(f"  note: {impl}: {result.error}")


def validate_args(args: argparse.Namespace) -> None:
    if "mamba2_simple" in args.families:
        for d_model in args.d_models:
            d_inner = args.expand * d_model
            if d_inner % args.headdim != 0:
                raise SystemExit(
                    f"Mamba2Simple requires expand*d_model divisible by headdim; "
                    f"got expand={args.expand}, d_model={d_model}, headdim={args.headdim}"
                )


def _safe_run(
    factory: Callable[[], BenchmarkResult],
    *,
    fallback: Callable[[str | None], BenchmarkResult],
    strict: bool,
) -> BenchmarkResult:
    try:
        return factory()
    except Exception as exc:
        if strict and not _is_oom(exc):
            raise
        tag = "OOM" if _is_oom(exc) else f"{type(exc).__name__}: {exc}"
        _reset_cuda_state()
        return fallback(tag)
    finally:
        _reset_cuda_state()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Benchmarking selective-read matrix with "
        f"batch_sizes={args.batch_sizes} dtypes={args.dtypes} "
        f"d_state={args.d_state} d_conv={args.d_conv} expand={args.expand} "
        f"headdim={args.headdim} ngroups={args.ngroups}"
    )
    print(f"families={','.join(args.families)} passes={','.join(args.passes)}")

    all_results: list[BenchmarkResult] = []
    for pass_name in args.passes:
        for dtype_str in args.dtypes:
            dtype = getattr(torch, dtype_str)
            for batch in args.batch_sizes:
                for d_model in args.d_models:
                    for seqlen in args.seq_lens:
                        shape_results: list[BenchmarkResult] = []
                        for family in args.families:
                            for selective_read in [True, False]:
                                def _fallback(err: str | None, *,
                                              family=family, selective_read=selective_read,
                                              pass_name=pass_name, dtype_str=dtype_str,
                                              batch=batch, seqlen=seqlen, d_model=d_model) -> BenchmarkResult:
                                    return BenchmarkResult(
                                        family=family,
                                        selective_read=selective_read,
                                        pass_name=pass_name,
                                        dtype=dtype_str,
                                        batch=batch,
                                        seqlen=seqlen,
                                        d_model=d_model,
                                        ms=None,
                                        toks_per_sec=None,
                                        peak_memory_mb=None,
                                        benchmark_group="matrix",
                                        error=err,
                                    )
                                result = _safe_run(
                                    lambda family=family, selective_read=selective_read: run_case(
                                        family=family,
                                        selective_read=selective_read,
                                        benchmark_group="matrix",
                                        pass_name=pass_name,
                                        dtype_str=dtype_str,
                                        batch=batch,
                                        seqlen=seqlen,
                                        d_model=d_model,
                                        d_state=args.d_state,
                                        d_conv=args.d_conv,
                                        expand=args.expand,
                                        headdim=args.headdim,
                                        ngroups=args.ngroups,
                                        device=device,
                                        dtype=dtype,
                                        warmup=args.warmup,
                                        rep=args.rep,
                                    ),
                                    fallback=_fallback,
                                    strict=args.strict,
                                )
                                shape_results.append(result)
                                all_results.append(result)
                        print_shape_block(
                            shape_results,
                            families=args.families,
                            pass_name=pass_name,
                            dtype_str=dtype_str,
                            batch=batch,
                            seqlen=seqlen,
                            d_model=d_model,
                        )
                        static_c_impl_results: list[BenchmarkResult] = []
                        if "mamba2_simple" in args.families:
                            for static_c_impl in ["legacy_reference", "reduced"]:
                                def _baseline_fallback(err: str | None, *,
                                              static_c_impl=static_c_impl,
                                              pass_name=pass_name, dtype_str=dtype_str,
                                              batch=batch, seqlen=seqlen, d_model=d_model) -> BenchmarkResult:
                                    return BenchmarkResult(
                                        family="mamba2_simple",
                                        selective_read=False,
                                        pass_name=pass_name,
                                        dtype=dtype_str,
                                        batch=batch,
                                        seqlen=seqlen,
                                        d_model=d_model,
                                        ms=None,
                                        toks_per_sec=None,
                                        peak_memory_mb=None,
                                        benchmark_group="static_c_baseline",
                                        static_c_impl=static_c_impl,
                                        error=err,
                                    )
                                result = _safe_run(
                                    lambda static_c_impl=static_c_impl: run_static_c_baseline_case(
                                        impl=static_c_impl,
                                        pass_name=pass_name,
                                        dtype_str=dtype_str,
                                        batch=batch,
                                        seqlen=seqlen,
                                        d_model=d_model,
                                        d_state=args.d_state,
                                        d_conv=args.d_conv,
                                        expand=args.expand,
                                        headdim=args.headdim,
                                        ngroups=args.ngroups,
                                        device=device,
                                        dtype=dtype,
                                        warmup=args.warmup,
                                        rep=args.rep,
                                    ),
                                    fallback=_baseline_fallback,
                                    strict=args.strict,
                                )
                                static_c_impl_results.append(result)
                                all_results.append(result)
                            print_static_c_impl_block(
                                static_c_impl_results,
                                pass_name=pass_name,
                                dtype_str=dtype_str,
                                batch=batch,
                                seqlen=seqlen,
                                d_model=d_model,
                            )

    if args.json_out is not None:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print()
        print(f"Wrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
