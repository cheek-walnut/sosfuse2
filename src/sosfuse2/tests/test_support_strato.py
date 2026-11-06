"""Host-only test loader: no engine, CuPy, geometry, or disk cache initialization.

Only the wrapper's unavailable relative imports are replaced. This deliberately
supports testing flags/layouts on the supplied partial repository, not transport.
"""
from sosfuse2 import paths
import ast
import contextlib
import types
from pathlib import Path
import numpy as np

C = ('GROUND_PHASE_BATCH', 'LANE_ATTEN', 'LANE_GROUND', 'LM_LUT', 'PHASE1_TRACE3',
     'PHASE2_NOBARRIER', 'POINT_NOINLINE', 'RAY_CONST_SHARED', 'SH_AUX', 'SPECIALIZE')
C2 = C + ('TRANSPORT_WAVE_LANES', 'SH_AUX_PAIR')
BYTE = ('STRATO_LANE_GROUND', 'STRATO_LANE_FIRSTVAL')
NEW = BYTE + ('STRATO_SH_AUX', 'STRATO_MIE_LUT',
              'STRATO_MIE_LUT_FINE')


def load_wrapper(path=None):
    path = Path(path) if path else (paths.PKG/'fused.py')
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.level)]
    module = types.ModuleType('_strato_host_wrapper')
    module.__file__ = str(path.resolve())
    module.__package__ = 'sosfuse2'
    import importlib.util
    spec=importlib.util.spec_from_file_location('_sh_layout', path.with_name('sh_layout.py'))
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    module.__dict__.update({name:getattr(layout,name) for name in
                           ('count','padded','workspace','source_size','kernel_source','combined_aux')})
    module.__dict__.update(F=np.float32,
        TIMERS=types.SimpleNamespace(bind=lambda xp: None),
        stage=lambda *args, **kw: contextlib.nullcontext())
    from sosfuse2 import kernel_config
    module.__dict__.update({name: getattr(kernel_config, name) for name in
                            ('KERNEL_FLAGS', 'STAGE1_FLAGS', 'KSPLIT_VARIANTS', 'resolve_config', 'deterministic')})
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module
