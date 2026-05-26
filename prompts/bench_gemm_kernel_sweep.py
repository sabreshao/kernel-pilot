#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


# Keep this script runnable from either the AITER repo root or this directory.
SCRIPT_DIR = Path(__file__).resolve().parent
AITER_ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if path != str(SCRIPT_DIR)]
if str(AITER_ROOT) not in sys.path:
    sys.path.insert(0, str(AITER_ROOT))

import torch  # noqa: E402
import triton  # noqa: E402

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.gemm_op_a4w4 import gemm_a4w4  # noqa: E402
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.tuned_gemm import gemm_a16w16  # noqa: E402
import aiter.ops.triton.utils._triton.arch_info as arch_info  # noqa: E402


DEFAULT_M_VALUES = [4096]
DEFAULT_N_VALUES = [4096]
DEFAULT_K_VALUES = [4096]

FP8_BLOCK_K = 128
FP8_BLOCK_N = 128
MXFP4_BLOCK_K = 32


@dataclass(frozen=True)
class PreparedCase:
    run: Callable[[], torch.Tensor]
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    a_scale_shape: tuple[int, ...] | None = None
    b_scale_shape: tuple[int, ...] | None = None
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    kernel: str
    M: int
    N: int
    K: int
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    tflops: float | None
    status: str
    a_shape: str = ""
    b_shape: str = ""
    a_scale_shape: str = ""
    b_scale_shape: str = ""
    notes: str = ""

    def as_csv_row(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "median_ms": "" if self.median_ms is None else f"{self.median_ms:.6f}",
            "min_ms": "" if self.min_ms is None else f"{self.min_ms:.6f}",
            "max_ms": "" if self.max_ms is None else f"{self.max_ms:.6f}",
            "tflops": "" if self.tflops is None else f"{self.tflops:.3f}",
            "status": self.status,
            "a_shape": self.a_shape,
            "b_shape": self.b_shape,
            "a_scale_shape": self.a_scale_shape,
            "b_scale_shape": self.b_scale_shape,
            "notes": self.notes,
        }


def parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated integer list: {value}") from exc


def shape_product(
    m_values: Sequence[int],
    n_values: Sequence[int],
    k_values: Sequence[int],
) -> list[tuple[int, int, int]]:
    return [(m, n, k) for m in m_values for n in n_values for k in k_values]


def normalize_kernels(kernels: Sequence[str]) -> list[str]:
    if "all" in kernels:
        return ["bf16", "fp8", "mxfp4"]
    return list(dict.fromkeys(kernels))


def dtype_max(dtype: torch.dtype) -> float:
    try:
        return float(torch.finfo(dtype).max)
    except TypeError:
        return float(torch.iinfo(dtype).max)


def nonzero_scale(scale: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(scale.dtype).tiny
    return torch.clamp(scale, min=tiny)


def quantize_fp8_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-row, per-128-K-block FP8 quantization."""
    if x.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    m, k = x.shape
    if k % FP8_BLOCK_K != 0:
        raise ValueError(f"K={k} must be divisible by {FP8_BLOCK_K} for FP8 1x128 quant")

    x_blocks = x.float().view(m, k // FP8_BLOCK_K, FP8_BLOCK_K)
    scale = nonzero_scale(x_blocks.abs().amax(dim=2) / dtype_max(dtypes.fp8))
    x_q = (x_blocks / scale.unsqueeze(-1)).view(m, k).to(dtypes.fp8)
    return x_q, scale


def quantize_fp8_128x128_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic 128x128 block FP8 quantization for stored weight [N, K]."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if n % FP8_BLOCK_N != 0 or k % FP8_BLOCK_K != 0:
        raise ValueError(
            f"N={n} and K={k} must be divisible by {FP8_BLOCK_N} and {FP8_BLOCK_K}"
        )

    n_blocks = n // FP8_BLOCK_N
    k_blocks = k // FP8_BLOCK_K
    w_blocks = weight.float().view(n_blocks, FP8_BLOCK_N, k_blocks, FP8_BLOCK_K)
    scale = nonzero_scale(w_blocks.abs().amax(dim=(1, 3)) / dtype_max(dtypes.fp8))
    weight_q = (w_blocks / scale[:, None, :, None]).view(n, k).to(dtypes.fp8)
    return weight_q, scale


def prepare_bf16(m: int, n: int, k: int, device: torch.device) -> PreparedCase:
    a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    # AITER linear-style GEMMs store B as [N, K], equivalent to logical [K, N].
    b = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    return PreparedCase(
        run=lambda: gemm_a16w16(a, b, bias=None, otype=torch.bfloat16),
        a_shape=tuple(a.shape),
        b_shape=tuple(b.shape),
        notes="A/B BF16, output BF16",
    )


def prepare_fp8_blockscale(m: int, n: int, k: int, device: torch.device) -> PreparedCase:
    a_src = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    b_src = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    a_q, a_scale = quantize_fp8_1x128(a_src)
    b_q, b_scale = quantize_fp8_128x128_weight(b_src)

    return PreparedCase(
        run=lambda: gemm_a8w8_blockscale(a_q, b_q, a_scale, b_scale, dtype=torch.bfloat16),
        a_shape=tuple(a_q.shape),
        b_shape=tuple(b_q.shape),
        a_scale_shape=tuple(a_scale.shape),
        b_scale_shape=tuple(b_scale.shape),
        notes=f"FP8={dtypes.fp8}, A block=1x128, B block=128x128, scales FP32",
    )


def prepare_mxfp4(m: int, n: int, k: int, device: torch.device) -> PreparedCase:
    if not arch_info.is_fp4_avail():
        raise RuntimeError(f"MXFP4 is not available on this architecture ({get_gfx()})")
    if k % MXFP4_BLOCK_K != 0:
        raise ValueError(f"K={k} must be divisible by {MXFP4_BLOCK_K} for MXFP4")

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    a_src = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    b_src = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    # gemm_a4w4's production path follows the existing preshuffled layout used in
    # op_tests/test_gemm_a4w4.py. The logical schema remains A/B block-32 E8M0.
    a_q, a_scale = quant_func(a_src, shuffle=True)
    b_q, b_scale = quant_func(b_src, shuffle=True)
    b_q = shuffle_weight(b_q, layout=(16, 16))

    return PreparedCase(
        run=lambda: gemm_a4w4(
            a_q,
            b_q,
            a_scale,
            b_scale,
            dtype=torch.bfloat16,
            bpreshuffle=True,
        ),
        a_shape=tuple(a_q.shape),
        b_shape=tuple(b_q.shape),
        a_scale_shape=tuple(a_scale.shape),
        b_scale_shape=tuple(b_scale.shape),
        notes=(
            f"MXFP4={dtypes.fp4x2}, scales={dtypes.fp8_e8m0}, "
            "block=32 along K, production preshuffle layout"
        ),
    )


def prepare_case(
    kernel: str,
    m: int,
    n: int,
    k: int,
    device: torch.device,
) -> PreparedCase:
    if kernel == "bf16":
        return prepare_bf16(m, n, k, device)
    if kernel == "fp8":
        return prepare_fp8_blockscale(m, n, k, device)
    if kernel == "mxfp4":
        return prepare_mxfp4(m, n, k, device)
    raise ValueError(f"unknown kernel: {kernel}")


def run_one(
    kernel: str,
    m: int,
    n: int,
    k: int,
    device: torch.device,
    warmup: int,
    rep: int,
) -> BenchmarkResult:
    try:
        torch.cuda.empty_cache()
        prepared = prepare_case(kernel, m, n, k, device)
        torch.cuda.synchronize()
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            prepared.run,
            warmup=warmup,
            rep=rep,
            quantiles=[0.5, 0.2, 0.8],
        )
        torch.cuda.synchronize()

        flops = 2.0 * m * n * k
        tflops = flops / median_ms * 1e-9
        return BenchmarkResult(
            kernel=kernel,
            M=m,
            N=n,
            K=k,
            median_ms=float(median_ms),
            min_ms=float(min_ms),
            max_ms=float(max_ms),
            tflops=float(tflops),
            status="ok",
            a_shape=str(prepared.a_shape),
            b_shape=str(prepared.b_shape),
            a_scale_shape="" if prepared.a_scale_shape is None else str(prepared.a_scale_shape),
            b_scale_shape="" if prepared.b_scale_shape is None else str(prepared.b_scale_shape),
            notes=prepared.notes,
        )
    except Exception as exc:  # Keep long sweeps running even if one kernel/shape is unsupported.
        return BenchmarkResult(
            kernel=kernel,
            M=m,
            N=n,
            K=k,
            median_ms=None,
            min_ms=None,
            max_ms=None,
            tflops=None,
            status="skipped",
            notes=f"{type(exc).__name__}: {exc}",
        )
    finally:
        torch.cuda.empty_cache()


def format_float(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def print_results(results: Iterable[BenchmarkResult]) -> None:
    columns = ["kernel", "M", "N", "K", "median_ms", "min_ms", "max_ms", "TFLOPS", "status"]
    rows = []
    for result in results:
        rows.append(
            [
                result.kernel,
                str(result.M),
                str(result.N),
                str(result.K),
                format_float(result.median_ms, 4),
                format_float(result.min_ms, 4),
                format_float(result.max_ms, 4),
                format_float(result.tflops, 2),
                result.status,
            ]
        )

    widths = [len(col) for col in columns]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    header = "  ".join(col.ljust(width) for col, width in zip(columns, widths))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].as_csv_row().keys()) if results else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_csv_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BF16, FP8 block-scale, and MXFP4 AITER GEMM kernels. "
            "Input generation and quantization are performed outside the timed region."
        )
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["fp8"],
        choices=["all", "bf16", "fp8", "mxfp4"],
        help="Kernel families to benchmark.",
    )
    parser.add_argument(
        "--m-values",
        type=parse_csv_ints,
        default=DEFAULT_M_VALUES,
        help="Comma-separated M values.",
    )
    parser.add_argument(
        "--n-values",
        type=parse_csv_ints,
        default=DEFAULT_N_VALUES,
        help="Comma-separated N values.",
    )
    parser.add_argument(
        "--k-values",
        type=parse_csv_ints,
        default=DEFAULT_K_VALUES,
        help="Comma-separated K values.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        nargs=3,
        type=int,
        metavar=("M", "N", "K"),
        help="Benchmark one explicit shape. Can be repeated. Overrides M/N/K grids.",
    )
    parser.add_argument("--warmup", type=int, default=25, help="triton.do_bench warmup count.")
    parser.add_argument("--rep", type=int, default=100, help="triton.do_bench repetition count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for input preparation.")
    parser.add_argument("--csv", type=Path, help="Optional path to write CSV results.")
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional limit on total (kernel, shape) cases, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ROCm device is not available; cannot run AITER GEMM benchmarks.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    kernels = normalize_kernels(args.kernels)
    shapes = [tuple(shape) for shape in args.shape] if args.shape else shape_product(
        args.m_values, args.n_values, args.k_values
    )
    cases = [(kernel, *shape) for kernel in kernels for shape in shapes]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    print(f"Device: {torch.cuda.get_device_name(device)} ({get_gfx()})")
    print(f"Cases: {len(cases)}")
    print(f"Warmup/rep: {args.warmup}/{args.rep}")

    results = [
        run_one(kernel, m, n, k, device, warmup=args.warmup, rep=args.rep)
        for kernel, m, n, k in cases
    ]
    print_results(results)

    if args.csv:
        write_csv(args.csv, results)
        print(f"Wrote CSV: {args.csv}")

    failures = [r for r in results if r.status != "ok"]
    if failures:
        print("\nSkipped/failed cases:")
        for result in failures:
            print(f"- {result.kernel} M={result.M} N={result.N} K={result.K}: {result.notes}")

    return 0 if any(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
