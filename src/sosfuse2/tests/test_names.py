"""Searchable names: no module-level function or class shares its name with a layer folder or a renamed module.

A search for a layer or module name should find that folder or module, not an unrelated function. The checked
definitions are the module-level def and class statements of the modules outside tests/ (methods and nested functions
are left out, names are compared case-sensitively). A definition inside the module of the same name, such as
pack_state.pack_state, is allowed.
"""

import ast

from sosfuse2.foundation import paths

# Modules that got a new name in 1.4.0.
RENAMED = (
    'available_cpus', 'content_identity', 'endpoint_contract', 'engine_core', 'extinction', 'flags',
    'incoming_quadrature', 'incoming_rings', 'kernel_diagnostics', 'ksplit_tolerance', 'numpy_reference',
    'preview_functions', 'quantized_table', 'radiance_table', 'ray_geometry', 'ray_geometry_cpu',
    'ray_geometry_parallel', 'run_state', 'shell_segments', 'solver_diagnostics', 'source_solver', 'strato_sh_aux',
    'stratosphere_checks', 'testkit', 'validation_evidence', 'zenith_interpolation',
)


def module_files():
    return [path for path in sorted(paths.PKG.rglob('*.py')) if path.relative_to(paths.PKG).parts[0] != 'tests']


def definitions(body):
    """Module-level def and class statements, including those under a module-level if, try, with or loop."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node
            continue
        for field in ('body', 'orelse', 'finalbody'):
            yield from definitions(getattr(node, field, []))
        for handler in getattr(node, 'handlers', []):
            yield from definitions(handler.body)


def test_renamed_modules_exist():
    stems = {path.stem for path in module_files()}
    assert len(set(RENAMED)) == len(RENAMED) and set(RENAMED) <= stems, sorted(set(RENAMED) - stems)


def test_no_definition_takes_a_layer_or_module_name():
    layers = {path.parent.name for path in paths.PKG.glob('*/__init__.py')} - {'tests'}
    taken = layers | set(RENAMED)
    clashes = []
    for path in module_files():
        own = path.parent.name if path.name == '__init__.py' else path.stem
        for node in definitions(ast.parse(path.read_bytes(), str(path)).body):
            if node.name in taken and node.name != own:
                clashes.append(f'{path.relative_to(paths.PKG)}:{node.lineno} {node.name}')
    assert not clashes, clashes
