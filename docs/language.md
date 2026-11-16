# Language

A Carbon kernel is a Python function that runs once per threadgroup. Python runs
once, at staging time, to build the kernel: ordinary Python helpers, lists and
`if` statements on constants work as usual, and runtime loops and branches become
device code.

```python
from carbon import language as cl


@cl.kernel
def scale(
    x: cl.buffer(cl.f16),
    output: cl.buffer(cl.f16),
    count: cl.u32,
    factor: cl.f32,
    BLOCK: cl.constexpr,
):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    value = cl.cast(cl.load(x, index, active, other=0.0), cl.f32)
    cl.store(output, index, cl.cast(value * factor, cl.f16), active)
```

## Parameters

| Annotation | Meaning |
| --- | --- |
| `cl.buffer(dtype)` | A device buffer of `f16`, `bf16`, `f32`, `u32`, `u8`, `i8`, `u16` or `i16` |
| `cl.buffer(dtype, rank=N)` | A buffer with a rank-N View bound at dispatch |
| `(cl.buffer(a), cl.buffer(b))` | Buffers bound in order as `name[0]`, `name[1]`; the body gets a tuple |
| `cl.u32`, `cl.f32` | A runtime scalar, passed as `scalars={...}` at dispatch |
| `cl.constexpr` | A compile-time value: a number, shape, `Layout`, `View`, atom... |

Each distinct set of constants compiles a separate kernel; runtime scalars and
bound View shapes do not. Kernel code can also read names from its enclosing
Python scope and index or slice compile-time values (`LAYOUT.shape[1]`,
`VIEW[0]`). Carbon reads those names when it stages the kernel, so pass anything
that changes between compiles as a `cl.constexpr`.

## Tiles and layouts

A tile is a value spread across the threads of a threadgroup. Its layout says
which SIMD group, lane and slot (per-thread register) holds each element.
`cl.arange(n)` and `cl.full(shape, value, dtype)` make tiles; both take
`layout=`. The default layout is striped: element i goes to thread i mod threads.

```python
from carbon.compiler import Layout

ROWS = Layout.blocked((8, 32), threads=64)  # each thread holds 4 neighbours
index = cl.arange(ROWS.size, layout=ROWS)
```

Operands of elementwise operations need the same shape and layout; scalars
broadcast. `cl.convert_layout(tile, LAYOUT)` moves values to new owners, and with
`free=True` it fails at staging unless every element stays in its thread.
`cl.reshape`, `cl.transpose` and `cl.stack` change coordinates without moving
data. Carbon needs full 32-lane SIMD groups: 32 to 1024 threads.

A stride of 0 in a layout's SIMD group or lane mode gives several threads the same
elements. `layout.across_groups(n)` gives each of `n` SIMD groups the whole tile,
for values every group needs:

```python
ROW = Layout.striped((64,), threads=32).across_groups(4)  # each SIMD group holds all 64
GRID = Layout((4, 64), Mode(4, 64), Mode(32, 1), Mode(2, 32))
x = cl.load(vector, cl.arange(64, layout=ROW), True, other=0.0)
row = cl.broadcast_to(x, (4, 64), axes=(1,), layout=GRID)  # no data moves
```

Loads and math run in every holder; stores and atomics come from one holder.
Reductions need tiles without repeated elements.

## Operations

| Operation | Notes |
| --- | --- |
| `cl.load(buffer, index, mask, other=...)` | Masked and out-of-bounds elements give `other` |
| `cl.store(buffer, index, values, mask)` | Masked and out-of-bounds elements are skipped |
| `+ - * /`, `// %`, `& \| ^ ~ << >>`, comparisons | f32 or u32; u32 wraps |
| `cl.cast(value, dtype)`, `cl.bitcast(value, dtype)` | Widen or narrow explicitly; bitcast keeps bits |
| `cl.exp`, `cl.exp2`, `cl.log`, `cl.log2`, `cl.sqrt`, `cl.rsqrt`, `cl.reciprocal` | f32, precise math |
| `cl.sin`, `cl.cos`, `cl.tan`, `cl.tanh`, `cl.sigmoid`, `cl.erf` | f32, precise math |
| `cl.fma(a, b, c)` | f32 `a * b + c` rounded once |
| `cl.abs`, `cl.floor`, `cl.ceil` | f32, exact |
| `cl.maximum(a, b)`, `cl.minimum(a, b)` | f32 or u32; f32 propagates NaN and orders -0 below +0 |
| `cl.select(cond, a, b)` | Evaluates both sides |
| `cl.sum`, `cl.max`, `cl.cumsum`, `cl.cummax` | Reductions and inclusive scans along an axis |
| `cl.broadcast_to(tile, shape, axes=...)` | Repeats values along new axes |
| `cl.item(tile)` | A one-element tile as a uniform scalar |
| `cl.atomic_add/min/max/store(buffer, index, values)` | Relaxed u32 atomics; f32 for add and store |
| `cl.program_id()`, `cl.simd_id()` | Threadgroup index, SIMD group in the threadgroup |

Arithmetic is f32 or u32: load `f16` or `bf16` storage, cast to f32, compute, and
cast back for the store. The integers `u8`, `i8`, `u16` and `i16` are storage too: a
cast extends them to u32 by their sign or converts them to f32 exactly, and a cast
from u32 truncates to them. Fast math and contraction are off; write `cl.fma` where a
fused multiply-add is wanted. Against the correctly rounded result, `sqrt`,
`reciprocal`, `fma`, `abs`, `floor` and `ceil` are exact, and the other functions
are within 1 to 4 ULPs. `sin`, `tan`, `tanh` and `abs` keep subnormal inputs; the
other functions flush them, as arithmetic does.
[`carbon.quantization`](../python/carbon/quantization.py) unpacks low-bit codes and
decodes FP8, FP6, FP4 and E8M0 values.

## Reductions and scans

`cl.sum(value, axis, scratch=None, order="fast")` lets the hardware choose the
addition order, so the last bits can depend on the layout. `order="pairwise"` adds
neighbours in a fixed tree and gives the same bits for any layout. `cl.max` returns
NaN if any input is NaN. A reduction that crosses SIMD groups needs a scratch tile
of the input's shape:

```python
shared = cl.scratch(TILE.shape, cl.f32)
total = cl.sum(values, 0, scratch=shared)
count = cl.item(cl.sum(counts, 0, scratch=shared), scratch=shared)
```

`cl.cumsum` and `cl.cummax` give inclusive running sums and maxima along an axis,
in the input's layout, with the same `order` choice. `cl.broadcast_to`,
`cl.convert_layout` and `cl.item` also need scratch when they move values between
SIMD groups. Carbon never allocates it for you.

## Control flow

`range` loops and `if`/`elif`/`else` on runtime scalars become device code, in
kernels and in `@cl.inline` helpers. A runtime condition must be the same for the
whole SIMD group; barriers and cross-SIMD collectives need the same value in the
whole threadgroup. Use masks, not `if`, for per-element choices. Values carried
through a loop keep their type and layout; initialize them before the loop.
`and`, `or` and `not` take compile-time operands, and `x if condition else y` a
compile-time condition. List comprehensions and generator expressions over
compile-time iterables unroll, eagerly, while the kernel stages:
`[cl.load(x, i * 32 + index, True, other=0.0) for i in range(PARTS)]` is `PARTS`
loads, and the list can cross runtime loops and branches as a tuple can. Python's
builtins, such as `range`, `len`, `zip` and `sum`, work as in any helper.
`while`, `break` and `continue` are not supported.

`cl.double_buffer(count, buffers=..., state=..., produce=..., consume=...)` runs a
two-stage schedule over a uniform count: the producer fills one stage of scratch
while the consumer reads the other and returns the next state. Corresponding stages
are distinct scratch allocations of one type.

## Views and partitions

A `View` describes strided storage: element type, shape, strides in elements, and
offset and extent in bytes. It allocates nothing.

```python
from carbon.compiler import View

MATRIX = View("f32", (64, 128), (128, 1), 64 * 128 * 4)

tile = cl.partition(cl.view(data, MATRIX), (16, 32), cl.program_id())
region = cl.subview(tile, (row, column), (rows, columns))
value = cl.load(tile, cl.arange(16 * 32), True, other=0.0)
```

`cl.partition` picks one tile of a row-major grid of tiles; `cl.subview` crops a
region. Both clip to the parent, and accesses outside are masked. `cl.dim(view,
axis)` reads a dimension: a partition's is its tile extent, even where the tile is
clipped at an edge, and a subview's is its clipped extent. Loads and stores take
flat indices or one coordinate tile per axis, `cl.load(region, (rows, columns),
True, other=0.0)`; coordinates need no division by runtime extents.

A ranked parameter, `cl.buffer(cl.f32, rank=2)`, takes its View at dispatch: pass
`descriptor.bind(buffer)`. Changing the shape or strides then needs no recompile.
`compile(..., views={"name": descriptor})` makes them constants instead; an extent
of `None` stays a runtime value while the strides, the other extents and the zero
offset become constants, so `View("f16", (None, 4096), (4096, 1), None)` serves
every row count. Tensor ops on a ranked parameter need this.

`cl.load` with a scalar index returns a scalar that can drive loops and branches,
for example `count = cl.load(lengths, cl.program_id(), True, other=0)`. The index
must be the same for the whole threadgroup, and the buffer read-only in the kernel.

## Memory

Every device load and store checks its mask, the tile's padding, and the bounds of
its buffer or View: a load out of range returns `other`, and a store does nothing.
Dispatch checks buffer sizes and alignment, and rejects overlapping buffers when
one of them is written. A check the compiler proves always true costs nothing; it
knows the range of each lane, SIMD group and loop counter, `max_grid` if you compile
with it, static Views, and these contracts:

- `compile(..., ranges={"offset": (0, 96)})` bounds runtime `u32` scalars; every
  dispatch rejects values outside them.
- `compile(..., min_bytes={"source": 512})` gives buffers a minimum size, which
  dispatch checks.

Neighbouring elements that one thread loads or stores become one vector access
(`float4`, `half4`...) wherever the compiler proves them contiguous and aligned.
`CopyAtom(dtype, width).layout(shape, threads=...)` gives each thread runs of
`width` neighbours to make that possible; `artifact.binding_alignment` says what
dispatch then requires.

### Writes

Which thread writes which element is up to the kernel: pages chosen through a
table, stores in loops, and atomics mixed with ordinary accesses all compile. The
kernel must make sure that two threads do not write one element in a dispatch, and
a thread sees another thread's device write only in a later dispatch.

`check_writes` runs a kernel once and reports elements that two threads wrote. This
kernel appends one block of keys per threadgroup to a cache page chosen by a page
table:

```python
from carbon.compiler import check_writes


@cl.kernel
def append(
    keys: cl.buffer(cl.f32),
    pages: cl.buffer(cl.u32),
    cache: cl.buffer(cl.f32),
    BLOCK: cl.constexpr,
):
    index = cl.arange(BLOCK)
    page = cl.load(pages, cl.program_id(), True, other=0)
    value = cl.load(keys, cl.program_id() * BLOCK + index, True, other=0.0)
    cl.store(cache, page * BLOCK + index, value, True)


# keys, pages and cache are device buffers filled as for a normal dispatch
conflicts = check_writes(
    append,
    device,
    [keys, pages, cache],
    constants={"BLOCK": 64},
    threads=64,
    grid=(8, 1, 1),
)
```

If two entries of `pages` are equal, `conflicts` holds
`WriteConflict(binding="cache", element=...)`, the lowest element that two threads
wrote; an empty list means none. Atomics and reads are not recorded.

Atomics are relaxed: they order nothing else, so read a finished table in a later
dispatch. f32 adds land in an order the hardware chooses.

### Scratch and barriers

`cl.scratch(shape, dtype, strides=None)` allocates threadgroup memory. Use
`cl.barrier()` between a write and a read by another thread, and again before you
overwrite data that other threads still read:

```python
shared = cl.scratch((ROWS, COLUMNS), cl.f32)
cl.store(shared, source_index, value, True)
cl.barrier()
value = cl.load(shared, target_index, True, other=0.0)
cl.barrier()
```

The compiler rejects a kernel that reads another thread's scratch write without a
barrier between them, reads scratch that no path has written, stores two elements
to one scratch address, or runs a barrier under control flow that differs between
threads. Allocations whose lifetimes a barrier separates share bytes, also inside
loops; `artifact.scratch_layout` shows the placement. A kernel may use all of the
device's threadgroup memory, or `compile(..., max_scratch=bytes)` of it.
