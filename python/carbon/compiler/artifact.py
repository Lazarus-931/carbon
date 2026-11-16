import struct
from dataclasses import dataclass

from ..language import Buffer, Scalar, u32
from .view import BoundView, View


@dataclass(frozen=True)
class ScratchRegion:
    """Placed scratch region; first/last are access positions, or -1 if unused.
    Regions share bytes only when their lifetimes are fenced apart."""

    offset: int
    bytes: int
    storage: str
    first: int
    last: int
    location: str


@dataclass(frozen=True)
class Artifact:
    """Compiled IR, Metal source and resource contract, checked by `dispatch`."""

    ir: str
    source: str
    access: tuple[str, ...]
    parameters: tuple[tuple[str, Scalar | Buffer], ...]
    threads: int
    llvm_version: str
    key: str
    compile_ms: float
    binding_bytes: tuple[int, ...]
    binding_alignment: tuple[int, ...]
    max_grid: int
    toolchain_id: str
    build_id: str
    scratch_bytes: int
    scratch_layout: tuple[ScratchRegion, ...] = ()
    views: tuple[tuple[str, View], ...] = ()
    checked_bindings: tuple[int, ...] = ()
    thread_ir: str = ""
    scalar_ranges: tuple[tuple[str, int, int], ...] = ()
    indirect: bool = False

    def dispatch(self, device, buffers, *, grid, scalars=None, checks=()):
        """Validate a launch and return its Dispatch without submitting it.
        The first dispatch to each Device compiles the Metal pipeline.

        Args:
            device: the Device to run on.
            buffers: one buffer or bound View per buffer parameter, in signature order.
                A runtime-view parameter takes a View bound to its buffer; each binding
                needs `binding_bytes` at an offset aligned to `binding_alignment`.
            grid: (threadgroups, 1, 1), with 1 to `max_grid` threadgroups.
            scalars: the runtime scalar parameters by name, within the compiled
                `scalar_ranges`.
            checks: a write-checked kernel's shadow and record buffers, as
                `carbon.compiler.check_writes` makes them.
        """
        scalars = {} if scalars is None else scalars
        buffer_parameters = [
            (name, kind) for name, kind in self.parameters if isinstance(kind, Buffer)
        ]
        scalar_parameters = [
            (name, kind) for name, kind in self.parameters if isinstance(kind, Scalar)
        ]
        buffers, views = self._bind_buffers(buffers, buffer_parameters)
        if set(scalars) != {name for name, _ in scalar_parameters}:
            raise ValueError("runtime scalar names do not match kernel signature")
        for name, minimum, maximum in self.scalar_ranges:
            value = scalars[name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be in the compiled inclusive range "
                    f"[{minimum}, {maximum}]",
                )
        indirect = None
        if not isinstance(grid, tuple) and self.indirect:
            indirect, grid = grid, (1, 1, 1)
        elif not isinstance(grid, tuple):
            raise ValueError(
                "a grid from a buffer needs a kernel compiled with indirect=True",
            )
        if (
            len(grid) != 3
            or grid[1:] != (1, 1)
            or any(type(n) is not int for n in grid)
            or not 1 <= grid[0] <= self.max_grid
        ):
            raise ValueError(
                "grid must be a one-dimensional tuple with 1 to "
                f"{self.max_grid} threadgroups",
            )
        if len(checks) != (
            len(self.checked_bindings) + 1 if self.checked_bindings else 0
        ):
            raise ValueError(
                "a write-checked kernel takes one shadow buffer per "
                "written binding and a record buffer",
            )
        for index, (buffer, minimum, alignment) in enumerate(
            zip(buffers, self.binding_bytes, self.binding_alignment),
        ):
            itemsize = buffer_parameters[index][1].element.itemsize
            extent = buffer.size if views[index] is None else views[index].extent_bytes
            if (
                buffer.offset % alignment
                or extent < minimum
                or buffer.size // itemsize > 2**32 - 1
            ):
                raise ValueError(
                    f"binding {index} needs {alignment}-byte aligned storage of at "
                    f"least {minimum} bytes within u32 indexing",
                )
            for other in range(index):
                if (
                    self.access[index] != "read" or self.access[other] != "read"
                ) and buffer.overlaps(buffers[other]):
                    raise ValueError(
                        "overlapping writable bindings are unsupported; pass the "
                        "storage as one binding",
                    )
        packed = self._pack_constants(
            scalars,
            scalar_parameters,
            buffers,
            views,
            buffer_parameters,
        )
        if "<metal_tensor>" in self.source and not device.tensor_ops:
            raise ValueError(
                "this kernel uses Metal 4 tensor ops, which need macOS "
                "26 and a Metal 4 GPU",
            )
        pipeline = device.compile(self.source, "carbon_kernel", fast_math=False)
        if pipeline.simd_width != 32:
            raise ValueError("compiled layout requires a 32-lane SIMD pipeline")
        return device.dispatch(
            pipeline,
            [*buffers, *checks],
            packed,
            grid=grid,
            threads=(self.threads, 1, 1),
            scratch_bytes=self.scratch_bytes,
            writes=[access != "read" for access in self.access] + [True] * len(checks),
            indirect=indirect,
        )

    def _bind_buffers(self, arguments, parameters):
        if len(arguments) != len(parameters):
            raise ValueError("buffer count does not match kernel signature")
        views, buffers = [], []
        specialized = dict(self.views)
        for argument, (name, kind) in zip(arguments, parameters):
            if name in specialized and specialized[name].runtime:
                if not isinstance(argument, BoundView) or not specialized[name].matches(
                    argument.descriptor,
                ):
                    raise ValueError(
                        f"{name} was compiled for {specialized[name]}; "
                        "bind a View that matches it",
                    )
            elif name in specialized:
                if isinstance(argument, BoundView):
                    if argument.descriptor != specialized[name]:
                        raise ValueError(
                            f"{name} was compiled for "
                            f"{specialized[name]}, not {argument.descriptor}",
                        )
                    argument = argument.buffer
                views.append(None)
                buffers.append(argument)
                continue
            if kind.rank is None:
                if isinstance(argument, BoundView):
                    raise TypeError("raw buffer parameters cannot receive a bound view")
                views.append(None)
                buffers.append(argument)
                continue
            if not isinstance(argument, BoundView):
                raise TypeError("ranked buffer parameters require View.bind(buffer)")
            view = argument.descriptor
            if (
                view.element != kind.element.name
                or len(view.shape) != kind.rank
                or view.address_space != "device"
            ):
                raise ValueError(
                    "bound view element type, rank, and address space "
                    "must match the parameter",
                )
            if (
                view.extent_bytes > argument.buffer.size
                or view.size >= 2**32
                or any(n >= 2**32 for n in view.shape)
                or any(not -(2**63) <= stride < 2**63 for stride in view.strides)
            ):
                raise ValueError(
                    "bound view needs valid storage, u32 dimensions, and i64 strides",
                )
            views.append(view)
            buffers.append(argument.buffer)
        return buffers, views

    @staticmethod
    def _pack_constants(scalars, scalar_parameters, buffers, views, buffer_parameters):
        """The constant block, the dispatch ABI the printer declares as `struct
        Constants`. Every field is four bytes: the runtime scalars in signature order,
        each binding's element count, then for each runtime view its extents, its
        strides as little-endian i64 (lo, hi) halves, and its offset in elements."""
        packed = bytearray()
        for name, kind in scalar_parameters:
            value = scalars[name]
            if kind == u32:
                if type(value) is not int or not 0 <= value < 2**32:
                    raise ValueError(f"{name} must be a u32")
                packed.extend(struct.pack("<I", value))
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be an f32 scalar")
                packed.extend(struct.pack("<f", value))
        for buffer, view, (_, kind) in zip(buffers, views, buffer_parameters):
            extent = buffer.size if view is None else view.extent_bytes
            packed.extend(struct.pack("<I", extent // kind.element.itemsize))
        for view in views:
            if view is not None:
                packed.extend(struct.pack(f"<{len(view.shape)}I", *view.shape))
                packed.extend(struct.pack(f"<{len(view.strides)}q", *view.strides))
                packed.extend(struct.pack("<I", view.offset_bytes // view.itemsize))
        return bytes(packed)
