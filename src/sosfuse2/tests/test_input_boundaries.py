"""Invalid budgets fail before work starts; dark preview centres remain renderable."""

import sys
import numpy as np
import pytest
from sosfuse2 import pipeline, table_fast
from sosfuse2.host.pack_state import render_functions


def test_preview_dark_centre():
    funcs = render_functions()
    rgb = np.zeros((100, 100, 3))
    rgb[0, 0] = 1
    key, exposure = funcs['central'](rgb)
    assert key > 0 and np.isfinite(exposure) and exposure > 0
    # A uniform positive centre follows the same exposure calculation.
    assert funcs['central'](np.ones_like(rgb)) == (key, exposure)
    with pytest.raises(ValueError, match='E_PREVIEW_LUMINANCE'):
        funcs['central'](np.zeros_like(rgb))


def test_pipeline_zero_threads(tmp_path):
    args = pipeline.parser().parse_args([
        '--states', str(tmp_path / 'missing.json'), '--ids', 'A',
        '--work-root', str(tmp_path), '--threads', '0',
    ])
    with pytest.raises(ValueError, match='E_PIPELINE_CPU_BUDGET'):
        pipeline.validate(args)


def test_table_nonpositive_batch(monkeypatch, tmp_path, capsys):
    for batch in (0, -1):
        with pytest.raises(ValueError, match='E_TABLE_BATCH'):
            table_fast.make_table(None, None, [], [], [], batch=batch)
        monkeypatch.setattr(sys, 'argv', [
            'table_fast', '--id', 'A', '--source', str(tmp_path / 'missing.npz'),
            '--prepare', str(tmp_path), '--out', str(tmp_path / 'table.npz'),
            '--batch', str(batch),
        ])
        with pytest.raises(SystemExit) as exc:
            table_fast.main()
        assert exc.value.code == 2 and 'E_TABLE_BATCH' in capsys.readouterr().err
        assert not (tmp_path / 'table.npz').exists()
