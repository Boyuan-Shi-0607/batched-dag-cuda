# batched-dag-cuda

C++17/CUDA kernels for evaluating the **same weighted, layered DAG over many
independent inputs**. Build and upload the topology once, then reuse it with
different weights. The project separates graph construction from execution;
a small diagrammatic-summation graph is included as a worked example.

For sample `b`, an edge contributes

```text
next[b, child] += coefficient * current[b, parent] * weights[b, weight_index]
```

Edges connect adjacent layers. A node sums its incoming contributions. Signed
coefficients, repeated edges, multiple sources, and outputs from intermediate
layers are supported. This is a specialized sum-product evaluator, not an
arbitrary task-graph scheduler: an edge cannot multiply two computed nodes.

## Kernels

| Backend | Work assignment | Accumulation |
| --- | --- | --- |
| `cpu` | Serial edge interpreter | CPU reference |
| `cuda-edge` | Threads visit `(sample, edge)` pairs | Global atomics |
| `cuda-shared` | Blocks visit `(sample group, partition)` pairs | Shared atomics, then global atomics |
| `cuda-gather` | One warp per `(sample, child)` | Register reduction, one store, no atomics |

The shared backend packs local edge indices into eight bytes and caches up to
four edges per thread in registers. Its partitions obey a shared-memory budget
computed from their actual local dictionaries. The gather backend uses a
child-oriented CSR layout. These are alternative mappings of the same graph;
the best choice depends on fan-in, reuse, batch size, and hardware.

All CUDA backends support FP32, FP64, and optional CUDA Graph replay. Each replay
starts from the original source values and resets its accumulation buffers.
The mathematical DAG and a CUDA Graph are different objects: the latter
records the per-layer kernel, initialization, and readout launches.

## Build and Test

Requirements: CMake 3.20+, a C++17 compiler, and, for GPU execution, CUDA Toolkit
12+ with a compatible NVIDIA driver/GPU. Select the CUDA architecture for your
device; `80` below targets A100. The default architecture is also `80`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBDG_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bdg_cuda_tests --require-device
```

The last command requires an accessible GPU. Ordinary CTest marks GPU tests as
**skipped**, not passed, when no device is available. A requested CUDA benchmark
never silently falls back to CPU.

CPU-only builds work without the CUDA toolkit, including on macOS:

```bash
cmake -S . -B build-cpu -DBDG_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

## Run a Known Example

```bash
./build-cpu/bdg_benchmark --graph data/diamond.dag \
  --input data/diamond.batch --backend cpu --outputs diamond-values.csv
```

The two output rows are `[29, -185]` and `[1, -19]`; the first column is an
intermediate-layer readout. Both the graph and its input file are short,
human-readable examples of the [file format](docs/format.md).

On a GPU, evaluate the included CoS graph with a deliberately non-divisible
batch size:

```bash
./build/bdg_benchmark --graph data/cos_su2_n2.dag \
  --backend cuda-shared --precision double --batch 13 --iterations 100 \
  --json shared.json
```

Use `cuda-edge` or `cuda-gather` with the same graph, seed, batch, and precision
for a controlled comparison. Use `--no-cuda-graph` to compare ordinary stream
launches. Every benchmark checks **all outputs** against a long-double CPU
reference before and after timing and fails if the check fails.

Generate a larger sample outside the timed evaluation:

```bash
./build/bdg_make_cos_fixture --order 4 --output cos-n4.dag
./build/bdg_benchmark --graph cos-n4.dag --backend cuda-gather \
  --precision double --batch 256 --iterations 100 --json gather.json
```

The generator accepts orders 1--8 with explicit size limits; high orders may
reach those limits. Its weights are synthetic unless an input file is supplied.
It is a kernel workload, not a complete physics simulation. See
[sample conventions](data/README.md) and [provenance](PROVENANCE.md).

## Reuse From C++

[examples/evaluate.cpp](examples/evaluate.cpp) shows loading a graph once and
evaluating two weight batches with the same executor. The public API has no
CUDA types in its headers. Input and output arrays are batch-major.

```cpp
bdg::CudaOptions options;
options.backend = bdg::CudaBackend::gather;
bdg::CudaExecutor<double> executor(graph, batch_size, options);
executor.set_inputs(weights); // Unit sources by default; optional sources argument.
executor.launch();
auto values = executor.download_outputs();
// values[b * graph.outputs.size() + output_index]
```

Call `set_inputs` again for a new batch with the same dimensions. It is a
blocking upload; `launch` enqueues work, and `download_outputs` synchronizes.
An executor owns its buffers/stream and must not be accessed concurrently.

Libraries can be installed with `cmake --install build --prefix <prefix>`.
Consumers use `find_package(bdg CONFIG REQUIRED)` and link `bdg::cuda` or
`bdg::core`. The source-tree sample generator is separate from those libraries.

## Documentation

- [Kernel design and memory layout](docs/kernel-design.md)
- [Benchmarking and profiling](docs/benchmarking.md)
- [Validation coverage and tested environments](docs/validation.md)
- [Measured order-eight SU(3) comparison with the original kernel](docs/order8-performance.md)
- [Graph and input formats](docs/format.md)
- [Contributing](CONTRIBUTING.md)

No universal speedup is claimed. GPU timing includes buffer initialization and
readouts but excludes topology construction and host/device transfers; these
costs matter when the graph or inputs cannot be reused. Atomic backends can
change summation order. See the validation and benchmarking notes before
using results in a scientific calculation.

## License

[MIT](LICENSE), copyright Boyuan Shi. See [CITATION.cff](CITATION.cff) for
software citation metadata.
