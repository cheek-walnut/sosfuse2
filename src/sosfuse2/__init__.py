"""Fused SOS transport: NVRTC CUDA and an operation-matched NumPy reference."""


def __getattr__(name):
    # The version is defined in foundation/version.py and read on first access, so `import sosfuse2` loads no submodule.
    if name == '__version__':
        from .foundation.version import __version__

        return __version__
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
