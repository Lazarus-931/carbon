<p align="center">
  <img src="https://github.com/user-attachments/assets/05a268fd-25d6-4296-8e0e-e3a91a53f060" width="96" height="96" alt="Carbon logo" />
</p>

<h1 align="center">Carbon</h1>

Carbon is a tile-oriented Python DSL for writing GPU kernels on Apple silicon. You
write a kernel over tiles of values, choose which thread owns each value with a
layout, and Carbon compiles it to Metal Shading Language. The core idea behind it is
to give much better and flexible control over fundamental layouts in the gpu such as the basic
8x8 thread group which carbon refers to as `Atom`.

With the changing inference landscape(speculative decode and larger/weirder model architectures),
better control of the hardware becomes crucial in maximizing good-put!

It's co-designed with [Swiss](https://github.com/Lazarus-931/swiss) and is inference-first!

This work asks whether model specific vs general inference engines debate is the right question, instead exploring if 
different primatives can help design a systel that utilizes model-specific optimizations with general purpose engine!
This comes down to, as we'll see, deeper hardware control!

We find that generally in Swiss, this better abstraction allows us to maximize efficiency in weird workloads mlx just wasn't designed for!

One of the major drives was model specific allows us to define model custom optimizations, with carbon, much of that
is still possible, with viewing the gpu in a algebraically way. With this, general engines remain maintainable and still
perform on-par with custom model engines!

Since the focus was exploring a DSL for the gpu, carbon doesn't have support for cpu. It's promising, not only to control cpu during
gpu works for things like drafter acceptances but also better kernel scheduling!


## Why

Fine-grained control on gpus is not ideal on apple silicon with mlx today. Delegating, holding and
releasing buffers, for a specific workload are not first class, and left much to developer to figure out.

In Carbon, it's easy as:

```python
from carbon import Device

device = Device()  # <- here we create the gpu itself (MTLCreateSystemDefaultDevice)
buffer = device.buffer(16)  # make buffer
buffer.write(bytes(range(16)))  # write to it
data = buffer.read()  # read it here
del buffer  # delete it
```

With this, there is much fine grained control over buffer cache and allocation which means developers can
allocate what they need and release what they don't.

Furthermore, with it's Algebraic Layout approach similar to cutlass. This allows us to define
abnormal shapes and sizes! It gives you the flexibility of defining custom layouts such as:

```python
from carbon.compiler import Layout, Mode

rows = Layout((4, 64), Mode(4, 64), Mode(32, 1), Mode(2, 32))
owner = rows.owner((3, 40))
coordinate = rows.coordinate(2, 7, 1)
same_owner = rows.transpose((1, 0)).owner((40, 3))

odd = Layout((3, 31), Mode(1, 0), Mode(((2, 4), 4), ((4, 8), 1)), Mode(3, 32))
size, capacity = odd.size, odd.capacity
```

The layout maps elements to groups, lanes and slots. For (3, 40), the flat index is
3*64 + 40 = 232 = 64*3 + 8 + 32*1, giving `Owner(group=3, lane=8, slot=1)`.

Under the hood sits carbonIR which follows up with further optimizations and improvements to layouts!
This is done in inspiration to [CAKE](https://arxiv.org/abs/2608.12629), where I do see the evolving role agents play in optimizing the
IR to fit the demands of newer hardware and model architectures. As an agent-driven component of carbon, we leave
the python DSL as a human understandable components, like a window, to the agent's ever-evolving changes to the IR.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ccc9fba-4492-4e56-a8ac-ae1ef3d142d6" width="960" alt="Python and AST staging through tile IR, thread IR, and Metal" />
</p>

## Example

```python
from carbon import language as cl


@cl.kernel
def affine(
    x: cl.buffer(cl.f32),
    y: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    scale: cl.f32,
    BLOCK: cl.constexpr,
):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    a = cl.load(x, index, active, other=0.0)
    b = cl.load(y, index, active, other=0.0)
    cl.store(output, index, a * scale + b, active)


artifact = affine.compile(constants={"BLOCK": 128}, threads=64)
print(artifact.source)
```

## Install

To install on macOS, you need Python 3.10+, CMake 3.23+, a C++20 compiler,
and [metal-cpp](https://developer.apple.com/metal/cpp/).

There is a prebuilt package as well!

## Documentation

- [Quick start](docs/quickstart.md): build, then write, run and inspect a kernel.
- [Language](docs/language.md): tiles, layouts, operations, control flow, views, memory.
- [Matrix multiplication](docs/matrix.md): matrix atoms, Neural Accelerators,
  quantized weights.
- [Compiler](docs/compiler.md): how the pieces fit, tests and benchmarks.

Kernels in [`carbon.kernels`](python/carbon/kernels/):
[matrix products](python/carbon/kernels/gemm/) (dense, GEMV and quantized, with the
[launch](python/carbon/kernels/gemm/launch.py) dispatcher and the runtime search that pick their kernels and the
[epilogues](python/carbon/kernels/gemm/epilogue.py) they fuse into their stores),
[attention](python/carbon/kernels/attention/) over pages or one array (with the
[launch](python/carbon/kernels/attention/launch.py) dispatcher and the runtime search that picks its kernels), and
[small ops](python/carbon/kernels/ops/) that these use or that run alone:
[token sampling](python/carbon/kernels/ops/sampling.py) and
[row gathers and slot copies](python/carbon/kernels/ops/indexing.py). Their hooks share one
[`Hook`](python/carbon/kernels/common.py) class.

Examples: [dot products](examples/dot.py),
[elementwise ops on runtime views](examples/elementwise.py) and
[an MLP forward pass in one dispatch](examples/persistent_mlp.py).

## License

[MIT](LICENSE).
