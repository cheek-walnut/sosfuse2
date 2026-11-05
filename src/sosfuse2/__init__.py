"""Fused SOS transport: NVRTC CUDA and an operation-matched NumPy reference."""

# Install only when geometry is imported: controllers still need stdlib-only
# package imports. geometry.py itself is hashed into existing cache signatures.
import importlib
import sys
import types


class _Package(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'geometry' and __name__+'.geometry_parallel' not in sys.modules:
            importlib.import_module('.geometry_parallel', __name__)


sys.modules[__name__].__class__ = _Package
