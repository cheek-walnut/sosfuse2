"""SOSFUSE2_MLX=1 without the mlx package stops with E_MLX_UNAVAILABLE instead of an import traceback."""

import builtins

import pytest

from sosfuse2.backends import mlx_backend


def test_missing_mlx_is_an_explicit_error(monkeypatch):
    real_import = builtins.__import__

    def without_mlx(name, *args, **kwargs):
        if name == 'mlx' or name.startswith('mlx.'):
            raise ImportError(f'No module named {name!r}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_mlx)
    with pytest.raises(RuntimeError, match='E_MLX_UNAVAILABLE'):
        mlx_backend.require_mlx()


def test_mlx_switch_reads_only_one(monkeypatch):
    for value, expected in (('1', True), ('', False), ('0', False), ('yes', False)):
        monkeypatch.setenv(mlx_backend.ENV, value)
        assert mlx_backend.enabled() is expected
