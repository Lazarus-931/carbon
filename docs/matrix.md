# Matrix multiplication

Carbon has two matrix hardware operations: matrix atoms (Metal's 8×8 SIMD-group
matrices) and tensor ops (Metal 4 `matmul2d`, on the Neural Accelerators of M5). The
kernels in [`carbon.kernels.gemm`](../python/carbon/kernels/gemm/) are written in
Carbon like any other kernel; treat them as starting points.

## Matrix atoms

`Atom(dtype)` is one 8×8×8 multiply-accumulate per SIMD group, with f16, bf16 or
f32 operands and an f32 accumulator. Its operands are ordinary tiles: each SIMD
group holds one 8×8 block in the registers the device puts it in. Carbon measures
that layout on the device the first time a kernel needs it.

```python
from carbon.compiler import Atom

ATOM = Atom(cl.f16)  # pass as a constexpr
total = ATOM.zeros()
a = ATOM.load(left_view, left_tile, role="a")
b = ATOM.load(right_view, right_tile, role="b")
total = ATOM.mma(a, b, total)
ATOM.store(output_view, output_tile, total)
```

Loads take a rank-two View and the index of an 8×8 block: a number in row-major
order, or a (row, column) pair. Blocks past the edge read zeros. Operands are tiles,
so you can scale, cast, mask and store them like any other tile, or make them in
registers; `ATOM.layout(role, threads=...)` is the layout each role needs.
`TiledAtom(atom, groups, fragments)` arranges a grid of SIMD groups, each with a
grid of blocks, into one output tile.

`TensorAtom((m, n, k), dtype)` is the same model on the Neural Accelerators: one
m×n×k `matmul2d` per SIMD group on cooperative tensors, with f16 or bf16 operands
in registers. Use it in fused kernels such as attention, where one product feeds
more math before the next product.

## Kernels

Each module of `carbon.kernels.gemm` holds one family of kernels:

| Factory | File | What it does |
| --- | --- | --- |
| `make_tiled_matmul` | [tiled.py](../python/carbon/kernels/gemm/tiled.py) | TiledAtom tiles, batches, split-K; `make_sum_partials` sums the partials |
| `make_cooperative_gemv` | [gemv.py](../python/carbon/kernels/gemm/gemv.py) | One row times a matrix |
| `make_tensor_matmul` | [tensor.py](../python/carbon/kernels/gemm/tensor.py) | Neural Accelerators, see below |
| `make_qmv`, `make_qmm_registers`, `make_qmm`, `make_dequantize` | [quantized.py](../python/carbon/kernels/gemm/quantized.py) | Packed low-bit weights, see below |

[launch.py](../python/carbon/kernels/gemm/launch.py) runs one product.
`matmul(device, left, LEFT, right, RIGHT, output, OUTPUT)`, `dense_matmul` and
`quantized_matmul` list the plans that fit the call: `Tiles`, `Tensor` and `Vector`
for dense operands, and `QuantizedVector`, `QuantizedRegisters`, `QuantizedTiles` and
`Dequantized` for packed weights, with the measured rule's choice first. The runtime
search times them once per device and shape class and keeps the first unless another
is more than 3% faster. `CARBON_SEARCH=0` takes the first, and `plan=` names one.

## Epilogues

An epilogue is work fused into a product's store: the kernel hands it the f32 result
tile with each element's output row and column, and stores what it returns. Each
fused step saves a dispatch and a round trip through memory.
[epilogue.py](../python/carbon/kernels/gemm/epilogue.py) has `bias`, `relu`, `silu`,
`gelu`, `gelu_tanh`, `softcap`, `residual`, `multiply` and `rounded`, and `chain`
applies several in turn:

```python
from carbon.kernels.gemm import epilogue, matmul

fused = epilogue.chain(
    epilogue.bias(cl.f16), epilogue.gelu, epilogue.residual(cl.f16, columns=n)
)
steps = matmul(device, x, X, w, W, y, Y, epilogue=fused, operands=(b, r))
```

The buffers an epilogue reads, here the bias `b` and the residual `r`, bind after the
product's own, in order. `rounded(dtype)` rounds the product to `dtype` and back, so
`chain(rounded(cl.bf16), residual(cl.bf16, n))` adds the residual to the product as
a bf16 store would hold it. A `Glu` such as `epilogue.swiglu` pairs a gated product
before any epilogue: `quantized_matmul` takes a weight that stacks [gate | up]
halves, and `matmul` on matrix-atom tiles pairs neighbouring columns, 2j the gate and
2j + 1 the up of output column j, so its weight interleaves them; the output is half
as wide. Without one, SwiGLU takes two products: the gate projection stored through
`silu`, then the up projection with `multiply(dtype, n)` by the stored gate.
`residual` and `multiply` also take a `stride`, the elements between rows of a
wider matrix, so a product can read one slice of each row: bind the buffer at the
slice's first element, as `buffer.view(offset, size)`. For example, a gate by slice l
of rows L * P wide is `chain(rounded(cl.bf16), gelu_tanh, rounded(cl.bf16),
multiply(cl.bf16, P, stride=L * P))`, with the matrix bound at the slice's first element.
`softcap(c)` is c · tanh(x / c), a smooth cap to within ± c; it keeps the argmax.
Any function works: `Epilogue(function, operands=(cl.f16,))` calls
`function(value, row, column, *operands)`. Rows count across batches, so row r of
batch b is b * M + r. Elements past the output's edge, which the store drops, carry
coordinates outside it, so an epilogue reads memory but never writes it; its loads
are bounds-checked. `matmul`, `quantized_matmul` and the tiled, split-K, GEMV,
tensor-op and quantized factories take `epilogue=`; a split product applies it
where `make_sum_partials` adds the partials. With an epilogue, `make_tensor_matmul`
brings the product back to registers instead of letting the matrix unit write the
output.

## Neural Accelerators

You do not call the Neural Accelerators directly. On a device that has them
(`device.apple_family >= 10`, M5), the first plan of `matmul` is `Tensor` for
unbatched f16 and bf16 products with more than 8 rows and either M > 32 or N > 64,
and the first plan of `quantized_matmul` is `QuantizedTiles(tensor=True)` for f16 and
bf16 prefill with M > 32 or N > 1024. Everything else, f32 included, starts on the
matrix-atom kernels. On M4 the same tensor ops run on the shader cores; the first
plan takes them only for wide products (M ≥ 512 and N ≥ 1024, or M ≥ 4096 and
N ≥ 512, below a depth of 6144), where 64 x 64 tiles beat the matrix atoms by 2-6%.
The runtime search tries the tensor plans on any device that has the tensor ops.

Under the hood this is Metal 4's `matmul2d`:
`TensorMatmul(dtype).run(left, right, total)` in
[`make_tensor_matmul`](../python/carbon/kernels/gemm/tensor.py) has every SIMD group of
the threadgroup multiply two views that the hardware reads itself, and returns the
f32 product as a tile in `TensorMatmul(dtype).layout(shape, threads=...)`. Its
kernels need macOS 26, and dispatching one on a device without Metal 4 tensor ops
(`device.tensor_ops`) raises an error.

## Quantized weights

[quantized.py](../python/carbon/kernels/gemm/quantized.py) multiplies activations `x`
(M, K) by a packed weight (N, K): `output = x @ weight.T`, accumulated in f32. The
weight is stored row by row as `bits`-wide codes packed low bit first into u32
words. Each `group` neighbouring codes in a row share a scale, and for affine
formats a bias.

| Format | Weight | Scales |
| --- | --- | --- |
| `Affine(bits, group)` | `scale * code + bias` (MLX's layout) | Activation dtype |
| `Scaled(decode, bits, group)` | `decode(code) * scale` | E8M0 bytes, four per u32; activation dtype with `microscaled=False` |
| `MXFP4`, `MXFP8` | `Scaled` with FP4 E2M1 or FP8 E4M3 codes, group 32 | E8M0 bytes |

Any object with `bits`, `group`, `scale_type(dtype)`, `scale_bias(...)` and
`value(code)` works as a format.

```python
from carbon.kernels.gemm import Affine, quantized_matmul

steps = quantized_matmul(
    device,
    x,
    packed,
    scales,
    biases,
    output,
    rows=1,
    n=4096,
    k=4096,
    dtype=cl.f16,
    fmt=Affine(bits=4, group=64),
)
device.prepare(steps).submit().wait()
```

The first plan of `quantized_matmul` follows the rows: for one row, and for two rows of
codes that pack whole words, `QuantizedVector` (`make_qmv`), where one SIMD group computes each output feature and
each thread loads a word and its scale once for every row. Up to 32 rows take
`QuantizedRegisters` (`make_qmm_registers`) when eight groups of a row divide K, a group being one word or,
for codes that straddle words, the fewest words that hold a whole number of codes
(3 words and 16 codes at 6 bits, 3 and 32 at 3 bits, 5 and 32 at 5 bits). It unpacks
each lane's groups in registers for f32 matrix atoms, splits K over up to four SIMD
groups and multiplies blocks of eight rows; with the Neural Accelerators, only up to
eight rows do. Other shapes take `QuantizedTiles` (`make_qmm`), which dequantizes a weight
tile into scratch and multiplies with matrix atoms, or on the Neural Accelerators as
routed above.

With `glu=epilogue.swiglu`, a weight that stacks [gate | up] halves gives an output
of half the width, silu(gate) * up, from one dispatch on every route. The vector and
register kernels take each feature of both halves in the same SIMD group, so they
share one load of x; the tiles stage gate and up rows interleaved, so neighbouring
columns pair in the threads that hold them.

Given `scratch`, a buffer of n * k elements of the activation type, the first plan
for 512 rows or more off the Neural Accelerators is `Dequantized`: it dequantizes the
weight once, transposed, into the scratch and runs the dense product: on an M4, 0.99 times the tiles'
time at 512 rows and 0.92 at 2048. `make_dequantize` is that pass.

The packed layout matches MLX's `mx.quantize`.
