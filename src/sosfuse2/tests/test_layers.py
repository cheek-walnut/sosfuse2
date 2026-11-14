"""Layer rule: a module imports from its own layer or from a lower one.

The layers are the ten package folders in LAYERS, lowest first; modules at the package root sit above all of them.
Every import statement is read from the source, imports inside functions included, so a lazy import is a dependency
like any other (dynamic imports by name are not covered). The upward imports that remain are listed in UPWARD_IMPORTS
with the functions that hold them and the reason. The tests fail on an upward import that is not listed, on a listed
one that is gone, and on any upward import that runs when its module is imported. They check named modules and names
imported from packages, but do not check attributes imported from ordinary modules or execute the import statements.
"""

import ast
import importlib

from sosfuse2.foundation import paths

LAYERS = (
    'foundation', 'atmosphere', 'rays', 'precompute', 'backends', 'solve', 'outputs', 'jobs', 'checks', 'asset_tools',
)

# (importing module, imported module): (functions that hold the import, reason). A method is 'Class.method'.
UPWARD_IMPORTS = {
    ('sosfuse2.atmosphere.density', 'sosfuse2.rays.column_table'): (
        ('column_identity', 'exact_profile_columns'),
        'exact molecular column integrals and their cache identity come from the column table code',
    ),
    ('sosfuse2.atmosphere.density', 'sosfuse2.rays.ray_columns'): (
        ('attach',),
        'the exact solar-column correction replaces the per-component columns of a ray',
    ),
    ('sosfuse2.atmosphere.density', 'sosfuse2.precompute.host_cache'): (
        ('attach',),
        'attaching a profile reuses the cached corrected sun table and ground rays',
    ),
    ('sosfuse2.atmosphere.density', 'sosfuse2.asset_tools.generate_density_profile'): (
        ('asset_path',),
        'the density asset is generated into the work root when it is not there yet',
    ),
    ('sosfuse2.atmosphere.stratosphere', 'sosfuse2.precompute.ground_parallel'): (
        ('_ground_geometry',),
        'the stratospheric layer prepares its ground-cell geometry with the shared ground preparation',
    ),
    ('sosfuse2.atmosphere.stratosphere', 'sosfuse2.backends.sh_layout'): (
        ('attach', 'source_size', 'eigen_by_sh'),
        'spherical-harmonic array sizes',
    ),
    ('sosfuse2.atmosphere.tropo_mie', 'sosfuse2.backends.sh_layout'): (
        ('eigen_by_sh',),
        'spherical-harmonic array sizes',
    ),
    ('sosfuse2.rays.column_table', 'sosfuse2.precompute.host_cache'): (
        ('tables',),
        'column tables are stored under the CPU preparation cache keys',
    ),
    ('sosfuse2.solve.source_solver', 'sosfuse2.jobs.pipeline_execution'): (
        ('Solver.initialize',),
        'ray bundles are prepared for the kernel with the same code as in the persistent service',
    ),
    ('sosfuse2.solve.source_solver', 'sosfuse2.outputs.radiance_table'): (
        ('Solver.boxes_fixed',),
        'fixed observer boxes reuse the bundle and evaluation code of the radiance table',
    ),
    ('sosfuse2.outputs.radiance_table', 'sosfuse2.jobs.run_state'): (
        ('load_source', 'restored_settings', 'restored_table', 'main'),
        "restoring a stored source needs the runner's settings, contract version and state reader",
    ),
    ('sosfuse2.checks.testkit', 'sosfuse2.asset_tools.generate_minor_absorbers'): (
        ('synthetic_o4',),
        "the synthetic O4 test asset uses the generator's grid and record fields",
    ),
}


def modules(root):
    """Dotted name -> source path of every module outside tests/."""
    found = {}
    for path in sorted(root.rglob('*.py')):
        parts = ('sosfuse2', *path.relative_to(root).with_suffix('').parts)
        if parts[1] == 'tests':
            continue
        found['.'.join(parts[:-1] if parts[-1] == '__init__' else parts)] = path
    return found


def layer(module):
    parts = module.split('.')
    return LAYERS.index(parts[1]) if len(parts) > 1 and parts[1] in LAYERS else len(LAYERS)


def targets(node, package, found):
    """Modules named by one import; unresolved names imported from a package are reported as missing modules."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        base = node.module or ''
        if node.level:
            parts = package.split('.')
            base = '.'.join(parts[: len(parts) - node.level + 1] + ([node.module] if node.module else []))
        names = []
        is_package = base in found and found[base].name == '__init__.py'
        exports = vars(importlib.import_module(base)) if is_package else {}
        for alias in node.names:
            name = f'{base}.{alias.name}'
            if name in found or (is_package and alias.name != '*' and alias.name not in exports):
                names.append(name)
            else:
                names.append(base)
    else:
        return []
    return sorted({name for name in names if name.split('.')[0] == 'sosfuse2'})


def imports(root):
    """(importer, imported, function, line) per import statement; function is '' outside any function (a class body
    runs when the module is imported, so it counts as module level)."""
    found = modules(root)
    result = []

    def visit(module, package, node, scope, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested = in_function or not isinstance(child, ast.ClassDef)
                visit(module, package, child, (*scope, child.name), nested)
                continue
            for target in targets(child, package, found):
                result.append((module, target, '.'.join(scope) if in_function else '', child.lineno))
            visit(module, package, child, scope, in_function)

    for module, path in found.items():
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        visit(module, package, ast.parse(path.read_bytes(), str(path)), (), False)
    return found, result


def upward(root):
    found, result = imports(root)
    return found, [row for row in result if layer(row[0]) < layer(row[1])]


def where(found, root, rows):
    return [f'{found[a].relative_to(root)}:{line} {a} -> {b} in {f or "<module level>"}' for a, b, f, line in rows]


def test_layer_folders_are_the_packages():
    packages = {path.parent.name for path in paths.PKG.glob('*/__init__.py')} - {'tests'}
    assert packages == set(LAYERS)


def test_imported_modules_and_package_names_exist():
    found, result = imports(paths.PKG)
    missing = [row for row in result if row[1] not in found]
    assert not missing, where(found, paths.PKG, missing)


def test_upward_imports_are_the_listed_ones():
    found, rows = upward(paths.PKG)
    listed = {(a, b, f) for (a, b), (functions, _) in UPWARD_IMPORTS.items() for f in functions}
    unlisted = [row for row in rows if row[:3] not in listed]
    assert not unlisted, 'upward imports missing from UPWARD_IMPORTS: ' + '; '.join(where(found, paths.PKG, unlisted))
    gone = sorted(listed - {row[:3] for row in rows})
    assert not gone, f'UPWARD_IMPORTS entries with no such import left: {gone}'


def test_no_upward_import_at_module_level():
    found, rows = upward(paths.PKG)
    top = [row for row in rows if not row[2]]
    assert not top, where(found, paths.PKG, top)


def test_listed_entries_point_upward_and_give_a_reason():
    for (a, b), (functions, reason) in UPWARD_IMPORTS.items():
        assert layer(a) < layer(b) < len(LAYERS), (a, b)
        assert functions and '' not in functions and len(set(functions)) == len(functions), (a, b)
        assert reason.strip(), (a, b)
