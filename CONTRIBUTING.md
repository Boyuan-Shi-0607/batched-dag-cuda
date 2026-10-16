# Contributing

Keep topology construction, execution planning, and numerical evaluation
separate. A new backend must evaluate the same edge multiset and output list
as the CPU reference.

Before opening a change:

```bash
cmake -S . -B build-check -DBDG_ENABLE_CUDA=OFF \
  -DBDG_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-check -j
ctest --test-dir build-check --output-on-failure
```

For CUDA changes, also run all GPU tests with `--require-device`, and run
Compute Sanitizer as described in `docs/benchmarking.md`. CPU CI and CUDA
compile-only CI cannot establish GPU correctness.

Add focused tests for each bug fix. Preserve odd batch sizes, repeated replay,
changed inputs, signed coefficients, duplicate edges, and intermediate outputs.
Do not weaken tolerances to hide a discrepancy. Explain any precision-specific
failure and retain the input that reproduces it.

Performance reports should include GPU, driver/toolkit/compiler, build options,
graph, input seed or input file, batch size, precision, backend parameters, and
both validation and timing results. Compare identical workloads and separate
setup/transfer costs from device replay. Do not add private data, hostnames,
credentials, absolute user paths, build artifacts, or profiler dumps.

Contributions are made under the repository's MIT license.
