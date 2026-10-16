# Benchmarking and Profiling

## Measurement Contract

The CLI validates every output against a long-double CPU edge interpreter
before and after timing. Reference inputs and coefficients are first rounded
to the selected scalar type, so both paths receive the same mathematical
input. A comparison passes when `abs(actual-reference) <= atol + rtol*abs(reference)`;
nonfinite values fail. The CLI prints no throughput result on validation failure.
`long double` precision is platform-dependent and is not an arbitrary-precision
oracle. Integer-valued permutation tests provide an additional independent check.

For a zero reference, the reported relative-error field uses the absolute
error. An error statistic outside the representable `double` range is written
as JSON `null`, never an invalid `inf` token; validation still uses long-double
arithmetic and the stated tolerance rule.

CUDA event time includes the source reset, child initialization, per-layer
evaluation, and all requested readouts. It excludes plan construction,
allocation, input transfer, and output transfer. Host time measures the enqueue
loop through completion. Setup and upload times are reported separately.
For changing-input end-to-end workloads, measure upload, evaluation, and
download together in your application as well.

Reported logical edge throughput is

```text
GEdges/s = batch * original_edge_count / (device_ms * 1e6)
```

This is not a hardware FLOP counter. Generic coefficients may require an extra
multiply; gathers add reductions; atomics, resets, indexing, and readouts also
cost work. Zero-weight edges are still processed. No matrix-dependent pruning
is performed. Source-only graphs have zero logical edge throughput.

The CPU baseline is a serial reference interpreter that allocates buffers and
validates inputs per call. A ratio against this baseline is not a comparison
with an optimized CPU DAG implementation.

## Controlled Kernel Comparison

Run from the repository root after building for your GPU:

```bash
./build/bdg_make_cos_fixture --order 4 --output cos-n4.dag
mkdir -p results
for backend in cuda-edge cuda-shared cuda-gather; do
  ./build/bdg_benchmark --graph cos-n4.dag --backend "$backend" \
    --precision double --batch 256 --seed 1729 --warmup 10 --iterations 200 \
    --json "results/${backend}.json"
done
```

Repeat runs and report their distribution rather than selecting the best one.
Keep graph, inputs, precision, tolerances, clock/power conditions, and concurrent
GPU load fixed. The JSON records backend options, GPU, runtime/driver versions,
workspace size, timings, and errors. Also retain compiler version, CMake cache,
source revision, and the graph/input files or their hashes.

Small fixtures are primarily correctness tests. Kernel launch overhead can
dominate them. Increasing batch size changes both parallelism and memory use.
Do not interpret a short smoke run as a representative speedup study.

## Correctness Tools

```bash
./build/bdg_cuda_tests --require-device
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/bdg_cuda_tests --require-device --quick
compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./build/bdg_cuda_tests --require-device --quick
compute-sanitizer --tool synccheck --error-exitcode 1 \
  ./build/bdg_cuda_tests --require-device --quick
```

The quick mode runs 36 cases on small graphs, including a multi-partition
high-fan-in graph. The full mode adds empty layers, source-only outputs, CoS
order three, and more batch sizes. Tests evaluate changed inputs and repeated launches, not only the
first evaluation. Use a compute allocation on a cluster; do not execute GPU
tests on a login node. `examples/validate_gpu.pbs` is a short PBS example that
requires local scheduler/module configuration.

## Profiling

Configure with `-DBDG_CUDA_PTXAS_INFO=ON` to inspect register/spill reports.
Then profile a small number of iterations with ordinary stream launches:

```bash
ncu --set basic -o profile ./build/bdg_benchmark \
  --graph cos-n4.dag --backend cuda-shared --batch 256 \
  --precision double --warmup 0 --iterations 2 --no-cuda-graph
```

Inspect global/shared atomic traffic, memory load efficiency, occupancy,
barrier stalls, and register spilling. Profiled execution is not benchmark
timing. GPU counter access may require administrator configuration.

References: [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html),
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
