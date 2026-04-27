# Copyright (c) 2026 Sarthak Yadav

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable

import torch

try:
    from triton.testing import do_bench
except ImportError:
    do_bench = None

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2_simple import Mamba2Simple


FAMILY_DISPLAY = {
    "mamba": "Mamba",
    "mamba2_simple": "Mamba2Simple",
}


@dataclass
class BenchmarkResult:
    family: str
    selective_read: bool
    pass_name: str
    batch: int
    seqlen: int
    d_model: int
    ms: float | None
    toks_per_sec: float | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the 2x2 selective-read matrix for Mamba and Mamba2Simple "
            "over multiple sequence lengths and embedding dimensions."
        )
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--d-models", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--passes", choices=["forward", "forward_backward"], nargs="+", default=["forward", "forward_backward"])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
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


def run_case(
    *,
    family: str,
    selective_read: bool,
    pass_name: str,
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

    ms = benchmark_ms(fn, warmup=warmup, rep=rep)
    toks_per_sec = batch * seqlen / (ms / 1000.0)
    return BenchmarkResult(
        family=family,
        selective_read=selective_read,
        pass_name=pass_name,
        batch=batch,
        seqlen=seqlen,
        d_model=d_model,
        ms=ms,
        toks_per_sec=toks_per_sec,
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
    batch: int,
    seqlen: int,
    d_model: int,
) -> None:
    print()
    print(f"[{pass_name}] batch={batch} seqlen={seqlen} d_model={d_model}")
    print(
        f"{'family':<14} {'read=True ms':>12} {'read=False ms':>13} "
        f"{'false/true':>11} {'read=True tok/s':>16} {'read=False tok/s':>17}"
    )
    for family in families:
        family_results = {r.selective_read: r for r in results if r.family == family}
        read_true = family_results.get(True)
        read_false = family_results.get(False)
        read_true_toks = read_true.toks_per_sec if read_true is not None else None
        read_false_toks = read_false.toks_per_sec if read_false is not None else None
        print(
            f"{FAMILY_DISPLAY[family]:<14} "
            f"{format_metric(read_true.ms if read_true is not None else None):>12} "
            f"{format_metric(read_false.ms if read_false is not None else None):>13} "
            f"{format_speedup(read_false, read_true):>11} "
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


def validate_args(args: argparse.Namespace) -> None:
    if "mamba2_simple" in args.families:
        for d_model in args.d_models:
            d_inner = args.expand * d_model
            if d_inner % args.headdim != 0:
                raise SystemExit(
                    f"Mamba2Simple requires expand*d_model divisible by headdim; "
                    f"got expand={args.expand}, d_model={d_model}, headdim={args.headdim}"
                )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Benchmarking selective-read matrix with "
        f"dtype={args.dtype} batch={args.batch} d_state={args.d_state} "
        f"d_conv={args.d_conv} expand={args.expand} headdim={args.headdim} ngroups={args.ngroups}"
    )
    print(f"families={','.join(args.families)} passes={','.join(args.passes)}")

    all_results: list[BenchmarkResult] = []
    for pass_name in args.passes:
        for d_model in args.d_models:
            for seqlen in args.seq_lens:
                shape_results: list[BenchmarkResult] = []
                for family in args.families:
                    for selective_read in [True, False]:
                        try:
                            result = run_case(
                                family=family,
                                selective_read=selective_read,
                                pass_name=pass_name,
                                batch=args.batch,
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
                            )
                        except Exception as exc:
                            if args.strict:
                                raise
                            result = BenchmarkResult(
                                family=family,
                                selective_read=selective_read,
                                pass_name=pass_name,
                                batch=args.batch,
                                seqlen=seqlen,
                                d_model=d_model,
                                ms=None,
                                toks_per_sec=None,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        shape_results.append(result)
                        all_results.append(result)
                print_shape_block(
                    shape_results,
                    families=args.families,
                    pass_name=pass_name,
                    batch=args.batch,
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
