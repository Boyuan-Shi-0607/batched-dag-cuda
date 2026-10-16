# Validation

## Current Status

Checks performed during initial development on 2026-10-16:

| Environment | Check | Result |
| --- | --- | --- |
| macOS, AppleClang 15 | CPU Debug, AddressSanitizer + UndefinedBehaviorSanitizer | Six non-GPU CTests passed; GPU test explicitly skipped |
| macOS, AppleClang 15 | CPU Release, `-Wall -Wextra -Wpedantic` | Build without warnings; six non-GPU CTests passed |
| Linux, GCC 12.3, CUDA 12.1.105 | Release build targeting `sm_80` | All CUDA kernels and executables compiled and linked |
| Linux, GCC 12.3 | Non-GPU CTests from the CUDA build | Six passed |
| macOS CPU build and Linux CUDA build | Installed CMake package linked by a separate C++ consumer | Passed; CPU result checked |
| NVIDIA A100-SXM4-80GB, driver 580.126.20 | Full GPU suite, FP32/FP64, all three kernels, CUDA Graph on/off | All 288 cases passed |
| Same A100, CUDA 12.1 Compute Sanitizer | Quick suite under memcheck, racecheck, and synccheck | 36 cases passed per tool; zero errors; racecheck also reported zero warnings |
| Same A100 | Executor reuse with two changed weight batches | Both results matched the CPU reference |

The order-four sample (nine node layers, 5,176 edges) was also generated,
serialized, read back, and evaluated through the CPU benchmark with 13 inputs.
The no-CUDA fallback configuration, required-CUDA configuration failure, and
explicit CUDA request failure were checked on macOS.

The GPU validation job completed successfully on 2026-10-16 with exit status
zero and 21 seconds of scheduler-reported walltime. That duration includes the
test programs and sanitizer runs; it is not a kernel benchmark or a speedup
measurement. The GitHub Actions workflow is supplied but has not been run on
the hosted service.

The CUDA compiler's `ptxas` report for `sm_80` showed no stack frame or register
spill loads/stores in the compiled kernels. This is a compiler observation,
not a measurement of occupancy or speed. Other compiler versions and targets
can produce different code.

## Independent Checks

1. The diamond graph is checked against hand-calculated results, including a
   negative coefficient, multiplicity two, two non-unit sources, an intermediate
   output, and a final output. Both batch rows are checked.
2. For CoS orders 1--4, three integer-valued fixtures per order are checked
   against an independent enumeration of spin permutation pairs and all site
   assignments. Connectivity is tested on their union. Results are exact at
   these sizes, not accepted through a loose floating-point tolerance.
3. Packed partitions are decoded and their complete edge multiset is compared
   to the input graph. A CPU interpreter for those partitions is also compared
   to the ordinary edge interpreter for float/double-sized resource budgets,
   edge capacities 1, 7, and 385, and high-fan-in graphs.
4. Parser tests cover round trips, malformed/truncated/trailing input, negative
   unsigned fields, resource limits, invalid indices, nonfinite coefficients,
   and size-product overflow. The frozen CoS fixture is regenerated and compared.
5. Source-only graphs, empty edge layers, zero-indegree children, duplicated
   edges, and irregular batch/group sizes have dedicated checks.

## GPU Test Matrix

The full executable ran 288 cases on the A100:

- Six graphs: diamond, CoS order two, high fan-in, CoS order three, source-only,
  and an empty edge layer.
- Three CUDA backends, both FP32 and FP64, and CUDA Graph capture on/off.
- Batch sizes 1, 7, 13, and 33, including incomplete groups.

Each case uses three changed input batches: positive random, signed random,
and small exact integers. Every batch is replayed three times and checked,
then replayed again through the timing API and checked. Sources are non-unit.
Integer fixtures require exact equality. Random fixtures use mixed tolerances
(`1e-5 + 1e-4*abs(reference)` for FP32; `1e-11 + 1e-10*abs(reference)` for FP64).

Quick mode has 36 cases and retains the multi-partition high-fan-in graph.
The example PBS job runs the full suite, the reuse API example, and quick
Compute Sanitizer memcheck, racecheck, and synccheck tests. Every GPU execution
uses `--require-device` so a missing device fails the job instead of passing
through a skip.

## Interpretation

These checks cover the supplied implementation and fixtures. They do not prove
numerical accuracy for arbitrary ill-conditioned graphs. The atomic backends
may sum in different orders; the gather backend uses a different reduction
tree from the serial reference. Choose tolerances and precision appropriate
to the intended workload and retain the independent reference check when
adapting the kernels.

No physical simulation data or performance comparison with an optimized CPU
implementation is included in this initial validation. A separate
[order-eight FP32 comparison](order8-performance.md) measures the original CUDA
kernel against the new backends and reports numerical limitations as well as
speedups and regressions.
