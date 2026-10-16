# Order-eight SU(3) Performance Comparison

Measured on 2026-10-16. This compares the original shared-memory CUDA kernel
with the new executors on **one identical supplied DAG**, not different
mathematical summation algorithms. It includes slower results and numerical
limitations. No kernel or default selection was changed after observing these
measurements.

## Workload and Hardware

- GPU: NVIDIA A100-SXM4-80GB; driver 580.126.20.
- Build: GCC 12.3, CUDA 12.1.105, Release, target `sm_80`.
- GPU scalar: float (FP32) for every backend, including the original kernel.
- Model graph: the existing CoS-2 SU(N) uniform-shift generator with N_f=3,
  order eight, 47,259 nodes, 342,957 edges, 17 node layers, one source and sink.
- Maximum layer width: 8,520. No extra state pruning, edge merging, or
  input-dependent zero pruning was performed for this experiment.
- Dense seeded base matrices; nonzero main diagonals; matrix-diagonal shift
  `+0.137`. Only the true diagonal receives this shift, not the diagonal of an
  off-diagonal endpoint block.
- Two input seeds, 20261016 and 20261033, for numerical checks. The first seed
  is used for timing. Batch sizes: 1, 12, and 96.

The original CUDA kernel only supports signs +/-1. The SU(3) closure factor is
therefore represented by a precomputed `3*M_ij` entry and sign -1. Every backend
receives the same expanded float weight table, including the same rounded
closing weights. The graph is relabeled to layer-local IDs without changing
its edge multiset. Weight preparation is outside device timing for all routes.

Graph file SHA-256:
`46d505ee1a3a2547fd182089a07093f5f9debc5631b86eb7c328d70f1b36f0e5`.

## Baseline and Timing Scope

`legacy` uses a byte-identical copy of the original `evaluate_kernel.h`,
`preprocess_cpu.h`, `edge_structures.h`, and `utils.h` in a separate experiment
directory. The original HPC project was not modified. Kernel SHA-256:
`c27082b5ad70de9789a92d1036d5b81ad226f7ad040e15dff901e26095861298`.

The adapter retains the original 384-thread blocks, 1,536-edge/node partition
caps, partition ordering, level-dependent group schedule, and all-layer
numerical storage. Its driver includes resets in every replay, reads results
within bounds, checks CUDA errors, and covers partial groups. Thus the baseline
measures independent evaluations, not the old timing loop's accumulation over
stale values. This is a comparison to the original **kernel with a corrected
driver**, not a timing of the complete original physics application.

Primary timing uses CUDA events around repeated CUDA Graph launches. It covers
source reset, initialization, every propagation layer, and sink readout into
a device output buffer. Graph construction, upload, and host output transfer
are excluded. Seven rounds randomize backend order. Each round has ten warmup
evaluations, then 200 timed evaluations for batch one or 50 for larger batches.
The table reports the median round; all rounds and ranges are retained.

Clocks were not locked. Initial readings showed idle SM clocks, and some first
rounds were slower during warmup. No rounds were discarded. These are repeated
measurements within one allocation, not independent multi-day experiments.

## Resident Source-to-Sink Time

Times below are milliseconds for the **entire batch**, including the sink
readout. Speedup is legacy median divided by the corresponding new median.

| Backend | Batch 1 ms | Speedup | Batch 12 ms | Speedup | Batch 96 ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original kernel, corrected driver | 0.124006 | 1.000x | 0.269169 | 1.000x | 1.555395 | 1.000x |
| New edge, default parameters | 0.072801 | 1.703x | 0.147886 | 1.820x | 0.665743 | 2.336x |
| New shared, default parameters | 0.120387 | 1.030x | 0.242749 | 1.109x | 0.444211 | 3.501x |
| New gather, default parameters | 0.056013 | 2.214x | 0.243630 | 1.105x | 1.604936 | 0.969x |
| Shared, 384 threads / 1,536-edge cap | 0.121344 | 1.022x | 0.244408 | 1.101x | 0.425820 | 3.653x |
| Shared, one sample per group | 0.118554 | 1.046x | 0.149668 | 1.798x | 0.445911 | 3.488x |
| Shared, eight samples per group | 0.120207 | 1.032x | 0.400282 | 0.672x | 0.545976 | 2.849x |
| Shared, sixteen samples per group | 0.119675 | 1.036x | 0.562913 | 0.478x | 0.772137 | 2.014x |

The best measured choice depends on batch size: gather for one input, edge for
12 inputs, and the 384-thread shared configuration for 96 inputs. In particular,
gather is **3.19% slower** than legacy at batch 96. Increasing shared-group reuse
can also lose performance: group size 16 is more than twice as slow as legacy
at batch 12. These results do not justify one universally preferred backend or
claim that the small parameter sweep found an optimum.

At batch 96, default shared evaluates 74.12 billion logical edges/s; the
384-thread shared variant evaluates 77.32 billion logical edges/s. Its
amortized cost is 4.436 microseconds per sample. This is a throughput-derived
number, **not** a single-input latency; measured batch-one gather latency is
56.013 microseconds.

Raw round data and ranges: [summary](benchmarks/order8/summary.csv),
[batch 1](benchmarks/order8/b1-timings.csv),
[batch 12](benchmarks/order8/b12-timings.csv),
[batch 96](benchmarks/order8/b96-timings.csv).

## Transfer-Inclusive Check

A secondary check times five host calls including input validation, upload,
evaluation, synchronization, and output download, but excludes graph setup.
At batch 96 it gives 2.923571 ms for legacy, 1.709631 ms for default shared,
and 1.696758 ms for shared384: approximately 1.71x and 1.72x, respectively.
This supplementary check has fewer repetitions and fixed backend order, so it
is less robust than the randomized resident-device experiment. It establishes
that the 3.65x resident speedup must not be advertised as end-to-end speedup.

Raw data: [batch 1](benchmarks/order8/b1-endtoend.csv),
[batch 12](benchmarks/order8/b12-endtoend.csv),
[batch 96](benchmarks/order8/b96-endtoend.csv).

## Bandwidth: Measured and Unavailable

The [NVIDIA A100 specification](https://www.nvidia.com/en-us/data-center/a100/)
lists 2,039 GB/s for the 80-GB SXM model. Device-reported memory clock and bus
width imply 2,039.04 GB/s. A separate coalesced copy test, using a 256-MiB input
and 256-MiB output with pseudorandom data, measured 1,427.4--1,428.8 GB/s counting
one read and one write, about 70.0% of that peak. This is the copy kernel's
bandwidth, not the DAG's bandwidth.

**Actual DAG DRAM bandwidth and HBM utilization remain unavailable.** Nsight
Compute was invoked for complete CUDA Graphs with cache flushing and clock
locking disabled, requesting DRAM read/write bytes, duration, DRAM throughput,
L2 hit rate, and SM activity. All four profiling attempts failed with
`ERR_NVGPUCTRPERM`: the cluster does not grant this account performance-counter
access. No counter value was fabricated or inferred from the FLOP rate. An
administrator must grant access, or the run must be repeated on a machine
where counters are available, to answer the actual-HBM-utilization question.

The CSV's `logical_global_GB_s_NOT_DRAM` field is a **source-level access model**,
including metadata loads, numerical loads, logical atomic read/modify/write,
resets, and readouts. At batch 96 it is 134.3 GB/s for legacy, 1,609.8 for edge,
526.6 for shared, 466.8 for gather, and 519.9 for shared384. These numbers neither
account for cache hits nor measure memory transactions; dividing them by
2,039 to claim HBM utilization would be incorrect.

Similarly, `logical_2E_GFLOPs` uses two operations per edge as a common logical
work measure, not a hardware instruction count. Shared384 gives 154.64 logical
GFLOP/s under that convention. FLOP/s and bytes/s are different quantities.

The device has 40 MiB of L2. The measured per-executor allocation increments
were about 12--31 MB at batch 96. Cache reuse is therefore plausible, but L2
hit rate was not measured. The current evidence does **not** establish that
these runs saturate HBM or are limited specifically by HBM bandwidth. Atomic
contention, instruction/latency costs, insufficient parallelism, and launch
overhead remain candidates; assigning a bottleneck requires profiling.

## Memory Footprint

At batch 96, legacy's numerical workspace is 18,541,440 bytes, versus 6,937,344
bytes for each new executor, a 62.6% reduction. This includes weights, original
sources, propagation buffers, and outputs, but excludes topology/runtime
overhead. The reduction comes primarily from two ping-pong layer buffers
instead of retaining every layer. This is not a claim that total GPU memory
falls by the same percentage or that arbitrarily large batches cannot fail
allocation.

The legacy adapter also passed Compute Sanitizer memcheck on the order-eight
graph with batch 12: zero reported errors.

## Numerical Accuracy

Every sink was compared against a CPU long-double evaluation of the same
prepared float inputs, including both input seeds and repeated replay. The
predeclared check was

```text
abs(error_i) <= 1e-6 * max_j(abs(reference_j)) + 1e-3 * abs(reference_i)
```

All configurations passed that **mixed, batch-scaled** threshold, including
checks after timed replay. It does not guarantee 0.1% relative accuracy for
each output. The preflight data at batch 96 contain 192 outputs per backend;
one cancellation-sensitive output exceeds 0.1% relative error for every
backend. Its reference is approximately -0.1477966450:

| Backend | Observed float value | Relative error |
| --- | ---: | ---: |
| Legacy | -0.1487197876 | 0.6246% |
| Edge | -0.1496276855 | 1.2389% |
| Shared | -0.1531448364 | 3.6186% |
| Gather | -0.1503906250 | 1.7551% |
| Shared384 | -0.1495046616 | 1.1557% |

These are observations from the two-seed preflight, not worst-case error bounds
over all inputs or all atomic orderings. The other 191 outputs in each of these
preflight sets have relative error at most 0.1%. Atomic sums can vary with
execution ordering. The speed results establish execution performance at FP32;
they do not establish suitability for a requested physical accuracy target.

All per-sink values are retained: [accuracy summary](benchmarks/order8/accuracy-summary.csv),
[batch 1](benchmarks/order8/b1-sinks.csv),
[batch 12](benchmarks/order8/b12-sinks.csv),
[batch 96](benchmarks/order8/b96-sinks.csv).
