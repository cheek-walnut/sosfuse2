"""Import-order and recorded generator-identity checks."""

import ast
import subprocess
import sys

import pytest

from sosfuse2.foundation import paths


@pytest.mark.parametrize('first', ['ray_geometry', 'ray_geometry_parallel'])
def test_parallel_adapter_import_order(first):
    code = (
        f'import sosfuse2.solve.{first}\n'
        'from sosfuse2.solve import ray_geometry, ray_geometry_parallel\n'
        'assert ray_geometry.PackedCache is ray_geometry_parallel.PackedCache\n'
    )
    subprocess.run([sys.executable, '-B', '-W', 'error::ImportWarning', '-c', code], check=True)


def test_o4_generator_module_is_historical_literal():
    source = paths.PKG / 'asset_tools/generate_minor_absorbers.py'
    tree = ast.parse(source.read_bytes())
    common = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'common')
    keywords = [node for node in ast.walk(common) if isinstance(node, ast.keyword) and node.arg == 'module']
    assert len(keywords) == 1
    assert isinstance(keywords[0].value, ast.Constant)
    assert keywords[0].value.value == '.'.join(('sosfuse2', 'generate_minor_absorbers'))
