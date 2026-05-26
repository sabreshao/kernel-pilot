# mi300 block_scaled_mm KernelPilot Prompt

Copy this as one end-to-end prompt.

```text
/humanize:humanize-kernel-agent-loop

Use the local mi300 GPU environment for all mi300 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and rocprof Compute commands must run inside
the existing sabre_test Docker container, with GPU0 selected.

Use this command pattern for remote execution:

docker exec sabre_test bash -lc "HIP_VISIBLE_DEVICES=0 <command>"'

Do not run Python, pip, nvcc, builds, tests, benchmarks, or profiling directly
on the local bare metal.

Task:
Optimize block_scaled_mm kernel on AMD MI300x for this focused case:
You can run prompts/bench_gemm_kernel_sweep.py.

Target:
Beat the current aiter implementation by at least 1.10x on median latency for
the exact same shape, dtype, layout, bias behavior, and MI300x GPU0 environment.

Scope:
- Work in the current standalone workspace root. Do not create a nested repo
  unless the current directory is not writable.
- Build a benchmarkable and profileable CUDA/C++ or CUDA inline-PTX candidate.
- Keep the optimization focused on this single shape first.
- Do not change aiter behavior or public APIs unless a minimal local harness
  requires it for baseline measurement.

Baseline:
- Inspect the current aiter implementation path for int8_scaled_mm.
- Build a reproducible baseline harness before optimizing.
- Report aiter baseline latency for the exact focused case.
- Optional secondary baselines are allowed, such as torch._int_mm or fp16 GEMM,
  but the acceptance target is relative to aiter.

Correctness:
- Compare against the current aiter result and a PyTorch reference when
  practical.
- Include bias in the validation path.
- Report max absolute error, relative error, and the tolerance used.
- The final candidate must pass correctness before benchmark claims count.

Benchmarking:
- Use warmup and repeated timing.
- Report median latency, mean latency, std, min, p10, p90, and speedup over
  the aiter baseline.
- Keep benchmark scripts and raw result logs in the workspace.
- Every claimed improvement must identify the candidate commit/file version and
  the command used to produce the result.

Optimization guidance:
- Use KernelWiki when prior MI300x, gfx942, composable kernel, aiter, or block scale GEMM evidence
  is useful.
- Use Nsight Compute evidence when a candidate is correct but not clearly
  target-complete.
- Consider MI300x/gfx942 paths such as persistent scheduling, Stream-K or split-K,
  cluster shape choices, vectorized loads/stores, shared-memory staging, and a
  fused bias/output epilogue.
- Prefer evidence-backed edits over broad rewrites. Keep a performance map of
  tested variants and rejected ideas.

Completion:
- Continue iterating until the final correct candidate is at least 1.10x faster
  than the baseline on the focused case, or until at least six substantial
  evidence-backed attempts show why the target is blocked.
- The final report must include baseline numbers, final numbers, speedup,
  correctness tolerances, build/test/benchmark commands, key design decisions,
  and the next most promising follow-up if the target is not reached.
```
