"""Import-order and recorded generator-identity checks."""

import ast
import subprocess
import sys
from types import ModuleType

import pytest

from sosfuse2.foundation import paths
from sosfuse2.checks import testkit


@pytest.mark.parametrize('first', ['ray_geometry', 'ray_geometry_parallel'])
def test_parallel_adapter_import_order(first):
    code = (
        f'import sosfuse2.solve.{first}\n'
        'import sys\n'
        "ray_geometry = sys.modules['sosfuse2.solve.ray_geometry']\n"
        "ray_geometry_parallel = sys.modules['sosfuse2.solve.ray_geometry_parallel']\n"
        'assert ray_geometry.PackedCache is ray_geometry_parallel.PackedCache\n'
    )
    subprocess.run([sys.executable, '-B', '-W', 'error::ImportWarning', '-c', code], check=True, env=paths.child_env())


def test_legacy_aliases_restore_modules_and_attributes(monkeypatch):
    import sosfuse2
    from sosfuse2.atmosphere import spectral

    old = ModuleType('old_spectral')
    monkeypatch.setitem(sys.modules, 'sosfuse2.spectral', old)
    monkeypatch.setattr(sosfuse2, 'spectral', old, raising=False)
    monkeypatch.delitem(sys.modules, 'sosfuse2.air_wavelength', raising=False)
    monkeypatch.delattr(sosfuse2, 'air_wavelength', raising=False)
    with pytest.raises(RuntimeError, match='restore aliases'):
        with testkit.legacy_module_aliases('spectral', 'air_wavelength'):
            assert sys.modules['sosfuse2.spectral'] is sosfuse2.spectral is spectral
            with testkit.legacy_module_aliases('spectral'):
                assert sys.modules['sosfuse2.spectral'] is spectral
            raise RuntimeError('restore aliases')
    assert sys.modules['sosfuse2.spectral'] is sosfuse2.spectral is old
    assert 'sosfuse2.air_wavelength' not in sys.modules
    assert not hasattr(sosfuse2, 'air_wavelength')


def test_missing_production_files_report_package_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, 'PKG', tmp_path)
    monkeypatch.setattr(paths, 'PRODUCTION_MODULES', ('jobs/__init__', 'solve/__init__'))
    with pytest.raises(RuntimeError) as error:
        paths.production_files()
    assert 'jobs/__init__.py' in str(error.value)
    assert 'solve/__init__.py' in str(error.value)


def test_o4_generator_module_is_historical_literal():
    source = paths.PKG / 'asset_tools/generate_minor_absorbers.py'
    tree = ast.parse(source.read_bytes())
    common = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'common')
    keywords = [node for node in ast.walk(common) if isinstance(node, ast.keyword) and node.arg == 'module']
    assert len(keywords) == 1
    assert isinstance(keywords[0].value, ast.Constant)
    assert keywords[0].value.value == '.'.join(('sosfuse2', 'generate_minor_absorbers'))
