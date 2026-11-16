__version__ = "0.2.0"


def Device():
    """The system default Metal device."""
    from ._metal import Device as MetalDevice

    return MetalDevice()


def __getattr__(name):
    if name in {"Buffer", "Pipeline", "Dispatch", "Executable", "Completion"}:
        from . import _metal

        return getattr(_metal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
