# File Formats

Files are whitespace-delimited text. `#` starts a comment through the end of
the line. Indices are zero-based. Integers must be nonnegative decimal tokens;
real numbers must be finite. Unknown versions, missing fields, and trailing
tokens after `end` are rejected. The parsers use the classic numeric locale.

## Graph: BDAG 1

```text
BDAG 1
weights 4
layers 3
widths 2 2 1
outputs 2
1 1
2 0
edges 6
0 0 0 0 1
0 1 0 1 -1
0 0 1 2 1
0 1 1 3 1
1 0 0 2 2
1 1 0 0 -1
end
```

- `weights K`: number of scalar input weights per sample, positive.
- `layers L`: number of node layers including the source layer, positive.
- `widths`: exactly `L` positive layer widths.
- `outputs O`: exactly `O` pairs `layer node`, in requested output order.
- `edges E`: exactly `E` records `layer parent child weight coefficient`.

An edge in layer `l` reads a parent in node layer `l` and contributes to a child
in node layer `l+1`. Its value is `(parent * weight) * coefficient`.
Incoming contributions sum; a child with no incoming edges is zero. Duplicate
edges are retained and counted independently. Output pairs can refer to any
layer and can repeat. Layer zero contains user-supplied sources, defaulting to
one. Outputs are snapshots, not additional contributions to propagation.

Node and weight IDs are 32-bit. Device offsets into flattened arrays are
64-bit. Allocation products are checked. The parser exposes configurable
`ParseLimits` (including total nodes and input weight count); these are defensive limits, not guarantees that a graph will
fit on a particular device. Only load graphs whose memory requirements you
can accommodate.

## Input Batch: BDAG_BATCH 1

```text
BDAG_BATCH 1
batch 2
weights 4
sources 2
values
3 5 7 11
1 2 3 4
roots
1 2
-1 1
end
```

`values` contains `batch * weights` numbers, and `roots` contains
`batch * sources` numbers. Both are **batch-major**: all entries of sample zero
come first. Dimensions must agree with the graph. The CLI converts values to
the selected precision and rejects overflow to infinity.

Without an input file, the CLI generates positive synthetic weights from the
specified seed and uses unit sources. It does not inspect those weights to
remove edges or otherwise change the workload.

## Result Layout

The C++ API returns `batch * outputs` scalars, addressed by
`result[b * output_count + output_index]`. `--outputs path.csv` writes columns
`sample,output,layer,node,value`. For the graph and batch above, the output rows
are `[29, -185]` and `[1, -19]`.
