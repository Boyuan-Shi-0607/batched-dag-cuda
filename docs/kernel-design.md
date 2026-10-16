# Kernel Design

## Separate Structure From Values

`Graph` stores adjacent-layer edges and an ordered output list. `compile_plan`
validates the topology and creates child-oriented CSR and partition-local
maps. `CudaExecutor` uploads the chosen representation once, allocates its
workspace, and optionally captures an evaluation sequence as a CUDA Graph.
Changing weights does not rebuild any of these structures.

The numerical workspace contains input weights, original sources, two
ping-pong layer buffers, and outputs. Its scalar count is

```text
batch * (weight_count + source_width + 2 * max_layer_width + output_count)
```

This excludes topology arrays and CUDA runtime/graph overhead. Each current
layer's requested outputs are copied before that layer's buffer is reused.
The evaluator does not retain every intermediate layer.

## Edge Atomic Baseline

A flattened grid-stride loop visits `(sample, edge)` pairs. Each thread loads
the parent and weight, multiplies the coefficient, and atomically adds to the
child. Edge metadata are stored as separate arrays. A stream-ordered memset
zeros children before every layer.

This mapping exposes edge parallelism but issues a global atomic per edge.
High fan-in can cause contention. Indirect parent and weight reads are not
necessarily coalesced even when metadata reads are contiguous.

## Shared Partitions

Each partition has local dictionaries of parents, children, weights, and
coefficients. An eight-byte `PackedEdge` holds four unsigned 16-bit dictionary
indices. Global IDs remain 32-bit; only local dictionaries use 16-bit indices.

The planner limits both edge count and

```text
sizeof(scalar) * (parents + children + weights + coefficients)
```

which is the partition's dynamic shared-memory use. The requested budget is
clamped to the device limit. Extra shared memory is opted into through
`cudaFuncSetAttribute` only when needed. There is no fixed GPU-specific shared
memory allocation.

A block owns a `(sample group, partition)` task. Each thread loads up to four
packed edges into registers once, then reuses them across that sample group.
Parent values, weights, and coefficients are loaded into shared memory; child
sums use shared atomics. After a block barrier, each local child is flushed
with one global atomic. Different partitions may share global children, so
these final atomics are necessary. All barriers are block-uniform, including
the final incomplete sample group.

Grouping uses ceiling division. Partition edge capacity is bounded by
`min(requested_cap, 4 * threads)`. Child groups are round-robin interleaved
inside a partition to distribute immediate atomic destinations. This does
not guarantee a particular performance benefit; measure contention and
register pressure on the target workload.

## CSR Gather

CSR groups incoming edges by child. One warp owns one `(sample, child)` row.
Its lanes traverse incoming edges with stride 32, accumulate in registers,
then combine partial sums with `__shfl_down_sync`. Lane zero writes the result.
Each child is written once, including zero-indegree children. This removes
atomics and child memset, but very short rows waste lanes and irregular rows
can be imbalanced. Input layout remains batch-major.

## Independent Replay

Every evaluation copies original sources into scratch buffer zero. Atomic
backends clear every destination layer. Readouts overwrite their output
columns. The captured CUDA Graph includes all those operations; repeating it
does not accumulate previous results. Input uploads happen outside capture
and use the same device addresses. Source-only graphs and empty edge layers
follow the same contract.

An executor owns a nonblocking CUDA stream. `set_inputs` is synchronous with
respect to that stream, `launch` enqueues, and output download synchronizes.
Destroying an executor waits for its outstanding work. There is no concurrent
mutation or asynchronous input-lifetime API.

## Scope and Next Experiments

The kernels implement real FP32/FP64 sum-product graphs. They do not implement
complex scalars, arbitrary node operators, non-adjacent edges, device-side
graph construction, or multi-GPU sharding. Longer-range dependencies must be
represented by adjacent-layer carry edges, with an input weight fixed to one.

Candidates for later profiling include a lane-per-sample layout, degree-binned
gather kernels, child-owned partitioning, persistent kernels, and asynchronous
staging where there is enough reuse to overlap transfers. Inline PTX is not
used here: it should follow evidence from generated code and profiling, not
replace a measured memory/synchronization bottleneck by assumption.

References: [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[CUDA runtime execution attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html),
[inline PTX](https://docs.nvidia.com/cuda/inline-ptx-assembly/).
