# Quick start

## Build

Carbon needs macOS on Apple silicon, Python 3.10+, CMake 3.23+, Apple's command-line
tools, [metal-cpp](https://developer.apple.com/metal/cpp/), and the exact LLVM/MLIR
22.1.7 build that [`compiler/toolchain.json`](../compiler/toolchain.json) locks
(Homebrew's `22.1.7_1` bottle). A newer `brew install llvm` is not enough; CMake
rejects a mismatched toolchain. The installed package links LLVM statically, so you
need the pinned LLVM only to build.

```sh
python3 -m venv .venv
export CARBON_METAL_CPP_DIR=/absolute/path/to/metal-cpp
.venv/bin/python -m pip install . \
  -Ccmake.define.CARBON_BUILD_COMPILER=ON \
  -Ccmake.define.MLIR_DIR=/opt/homebrew/Cellar/llvm/22.1.7_1/lib/cmake/mlir
```

The metal-cpp path is the directory that contains `Metal/Metal.hpp`. Use the same
virtual environment to build and to run. For development, install with
`-e '.[dev]'` instead of `.`.

## Write and run a kernel

Save this as `multiply.py`. Carbon reads the kernel's source, so define kernels in
a `.py` file, not in a REPL.

```python
import struct

from carbon import Device
from carbon import language as cl


@cl.kernel
def multiply(
    x: cl.buffer(cl.f32),
    y: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    BLOCK: cl.constexpr,
):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    a = cl.load(x, index, active, other=0.0)
    b = cl.load(y, index, active, other=0.0)
    cl.store(output, index, a * b, active)


device = Device()
count, block = 93, 128
x, y, output = [device.buffer(count * 4) for _ in range(3)]
x.write(struct.pack(f"<{count}f", *([2.0] * count)))
y.write(struct.pack(f"<{count}f", *([3.0] * count)))
artifact = multiply.compile(constants={"BLOCK": block}, threads=64, device=device)
dispatch = artifact.dispatch(
    device,
    [x, y, output],
    grid=((count + block - 1) // block, 1, 1),
    scalars={"count": count},
)
executable = device.prepare([dispatch])
executable.submit().wait()
assert struct.unpack(f"<{count}f", output.read()) == (6.0,) * count
```

Each threadgroup (program) handles one block of 128 elements with 64 threads; the
mask `active` handles the last, partial block. `grid` counts threadgroups, not
threads. `carbon.Device()` is the system's default Metal device.

The steps are:

| Call | Does | Repeat it when |
| --- | --- | --- |
| `kernel.compile(...)` | Stages Python, verifies the IR, emits MSL | constants or `threads` change |
| `artifact.dispatch(...)` | Checks buffers, compiles the Metal pipeline | buffers, scalars or grid change |
| `device.prepare([...])` | Keeps an ordered list of dispatches | the list changes |
| `executable.submit().wait()` | Runs the work and waits | every call |

Keep the first three out of your hot loop. Between completed submissions you may
rewrite input bytes. Call `wait()` before you read or rewrite a buffer that a
submission uses; it also reports errors. Dispatches in one `prepare` run in order,
so one kernel can produce data that the next one reads. A tuple of dispatches in
the list is a group that may run at once, such as the query, key and value
projections of one input; `prepare` refuses a group in which one dispatch writes
bytes that another binds.

Submissions to one device run in the order you submit them, and each sees every
write of the ones before it. The host may write or read a buffer only while no
submitted command buffer binds it; `write` and `read` raise otherwise, so wait
for the last submission that uses the buffer first.

A kernel compiled with `indirect=True` can take its grid from the device: pass a
buffer range that holds three u32 threadgroup counts as `grid`, and an earlier
dispatch can write those counts, for example a split count that follows a length.
Threadgroups at `max_grid` or past it then return at once, so the compiled bound
still holds whatever the device writes.

## Choose a layout

A layout says which thread owns each element of a tile. The default is striped:
consecutive elements go to consecutive threads; `Layout.blocked` gives each thread
neighbouring elements instead. This kernel sums each row of a matrix; each thread
first accumulates its own elements, then `cl.sum` combines them:

```python
@cl.kernel
def row_sums(
    data: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    columns: cl.u32,
    SCHEDULE: cl.constexpr,
):
    shared = cl.scratch(SCHEDULE.shape, cl.f32)
    local = cl.arange(SCHEDULE.size, layout=SCHEDULE)
    total = cl.full(SCHEDULE.shape, 0.0, cl.f32, layout=SCHEDULE)
    for start in range(0, columns, SCHEDULE.size):
        active = local < columns - start
        total = total + cl.load(
            data, cl.program_id() * columns + start + local, active, other=0.0
        )
    cl.store(
        output,
        cl.full((), cl.program_id(), cl.u32),
        cl.sum(total, 0, scratch=shared),
        True,
    )


SCHEDULE = Layout.blocked((256,), threads=128)
sums = row_sums.compile(constants={"SCHEDULE": SCHEDULE}, threads=128)
```

`Layout` comes from `carbon.compiler`. The sum crosses SIMD groups, so it needs
scratch. See [row-wise dot products](../examples/dot.py) for a complete version.

## Inspect the output

```python
staged = multiply.stage(constants={"BLOCK": 128}, threads=64)
print(staged.ir)  # the tile IR the compiler built, for reading
print(artifact.ir)  # verified and optimized tile IR
print(artifact.thread_ir)  # per-thread IR after integer facts, before Metal
print(artifact.source)  # generated Metal Shading Language
```

## If something fails

- `_compiler` does not import: build with `CARBON_BUILD_COMPILER=ON` and the pinned
  `MLIR_DIR`, and run from the same virtual environment. Read the original
  import error.
- A tile cannot drive an `if`: a runtime `if` needs a uniform scalar. Use masks for
  elementwise choices, or `cl.item` to turn a one-element tile into a scalar.
- View strides count elements; View offsets and extents count bytes.
- Output is wrong or changes between runs: two threads may write one element. Run
  [`check_writes`](language.md#writes).
