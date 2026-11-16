import hashlib
import json
from time import perf_counter_ns

from . import cache
from .artifact import Artifact, ScratchRegion
from .frontend import free_values, stage
from .native import THREADGROUP_MEMORY, load_compiler

RESULT_FIELDS = (
    "ir",
    "thread_ir",
    "source",
    "access",
    "binding_bytes",
    "binding_alignment",
    "max_grid",
    "checked_bindings",
    "scratch_bytes",
    "scratch_layout",
    "llvm_version",
)


def _typed_items(mapping):
    """Sorted items, each value paired with its type and its items' types, since 1 ==
    1.0 == True."""

    def typed(value):
        items = tuple(map(typed, value)) if isinstance(value, tuple) else value
        return type(value), items

    return tuple(
        sorted((name, typed(value)) for name, value in dict(mapping or {}).items()),
    )


class Kernel:
    """A `@cl.kernel` function; `stage` and `compile` cache their results for repeated
    arguments."""

    def __init__(self, function):
        self.function = function
        self._cache = {}
        self._staged = {}

    def stage(
        self,
        *,
        constants,
        threads,
        max_grid=None,
        views=None,
        ranges=None,
        min_bytes=None,
        indirect=False,
    ):
        """Stage the kernel into the compiler's IR without lowering it."""
        return stage(
            self.function,
            constants=constants,
            threads=threads,
            max_grid=max_grid,
            views=views,
            ranges=ranges,
            min_bytes=min_bytes,
            indirect=indirect,
        )

    def _stage_once(
        self,
        constants,
        threads,
        max_grid,
        views,
        ranges,
        min_bytes,
        indirect,
    ):
        options = dict(
            constants=constants,
            threads=threads,
            max_grid=max_grid,
            views=views,
            ranges=ranges,
            min_bytes=min_bytes,
            indirect=indirect,
        )
        key = (
            threads,
            max_grid,
            bool(indirect),
            *map(_typed_items, (constants, views, ranges, min_bytes)),
            free_values(self.function),
        )
        try:
            hash(key)
        except TypeError:
            return self.stage(**options)
        if key not in self._staged:
            self._staged[key] = self.stage(**options)
        return self._staged[key]

    @staticmethod
    def _native(compiler, key, staged, capacity, check_writes):
        path = getattr(compiler, "__file__", None)
        stored = (
            hashlib.sha256(f"{key}{cache.binary_identity(path)}".encode()).hexdigest()
            if path
            else None
        )
        result = cache.load("artifacts", stored) if stored else None
        if not isinstance(result, dict) or any(
            field not in result for field in RESULT_FIELDS
        ):
            result = dict(compiler.compile(staged.staging, capacity, check_writes))
            if stored:
                cache.store("artifacts", stored, result)
        return result

    def compile(
        self,
        *,
        constants,
        threads,
        device=None,
        max_grid=None,
        max_scratch=None,
        views=None,
        ranges=None,
        min_bytes=None,
        check_writes=False,
        indirect=False,
    ):
        """Stage, verify and lower a kernel into an Artifact.

        Results are cached in memory per set of arguments and on disk per compiler
        build; the first dispatch to a Device compiles the Metal pipeline.

        Args:
            constants: values for the `cl.constexpr` parameters, by name.
            threads: threads per threadgroup, a multiple of 32 up to 1024.
            device: the Device whose threadgroup memory bounds scratch; None assumes 32
                KiB.
            max_grid: the most threadgroups a dispatch will launch, which bounds
                `cl.program_id()`.
            max_scratch: a byte budget for scratch, at most the device's threadgroup
                memory.
            views: compile-time Views for ranked buffer parameters, by name, fixing
                their shapes and strides. An extent of None stays a runtime value;
                dispatch then takes a View bound to a buffer, and a fully static View
                also takes the plain buffer.
            ranges: inclusive (minimum, maximum) bounds for runtime u32 scalars, by
                name, checked at each dispatch. Equal endpoints specialize a scalar
                without changing the kernel's signature.
            min_bytes: minimum byte sizes for buffer parameters, by name, checked at
                each dispatch; larger buffers keep their actual bounds, dimensions and
                strides.
            check_writes: make every device store record its writer, for
                `carbon.compiler.check_writes`.
        """
        start = perf_counter_ns()
        staged = self._stage_once(
            constants,
            threads,
            max_grid,
            views,
            ranges,
            min_bytes,
            indirect,
        )
        compiler = load_compiler()
        capacity = threadgroup_memory(device)
        payload = json.dumps(
            [
                staged.source,
                staged.fingerprint,
                check_writes,
                compiler.llvm_version,
                compiler.abi_version,
                compiler.toolchain_id,
                compiler.build_id,
            ],
            allow_nan=False,
        ).encode()
        key = hashlib.sha256(payload).hexdigest()
        if key not in self._cache:
            result = self._native(compiler, key, staged, capacity, check_writes)
            self._cache[key] = Artifact(
                result["ir"],
                result["source"],
                tuple(result["access"]),
                staged.parameters,
                threads,
                result["llvm_version"],
                key,
                (perf_counter_ns() - start) / 1e6,
                tuple(result["binding_bytes"]),
                tuple(result["binding_alignment"]),
                result["max_grid"],
                compiler.toolchain_id,
                compiler.build_id,
                result["scratch_bytes"],
                tuple(ScratchRegion(**region) for region in result["scratch_layout"]),
                staged.views,
                tuple(result["checked_bindings"]),
                result["thread_ir"],
                staged.scalar_ranges,
                bool(indirect),
            )
        artifact = self._cache[key]
        check_scratch(artifact, capacity, max_scratch)
        return artifact


def threadgroup_memory(device):
    """The bytes of threadgroup memory `device` has; 32 KiB without a device."""
    return getattr(device, "max_threadgroup_memory", THREADGROUP_MEMORY)


def check_scratch(artifact, ceiling, max_scratch):
    if max_scratch is None:
        budget = ceiling
    elif type(max_scratch) is not int or not 0 < max_scratch <= ceiling:
        raise ValueError(
            "max_scratch must be a positive byte count up to the device "
            f"ceiling of {ceiling}",
        )
    else:
        budget = max_scratch
    peak = artifact.scratch_bytes
    if peak > budget:
        regions = ", ".join(
            f"{region.storage} at {region.offset} ({region.bytes} B)"
            for region in artifact.scratch_layout
        )
        source = (
            "max_scratch"
            if max_scratch is not None
            else "the device's threadgroup memory"
        )
        raise ValueError(
            f"kernel needs {peak} bytes of threadgroup scratch but "
            f"{source} allows {budget}: {regions}",
        )
