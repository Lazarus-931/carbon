THREADGROUP_MEMORY = 32 * 1024
"""Default threadgroup memory budget in bytes when no device is supplied."""


def load_compiler():
    try:
        from .. import _compiler
    except ImportError as failure:
        raise RuntimeError(
            "Carbon's native compiler could not be loaded. Rebuild with "
            "CARBON_BUILD_COMPILER=ON and the pinned MLIR_DIR; see docs/quickstart.md. "
            f"Import error: {failure}",
        ) from failure
    return _compiler
