"""Required CPU regression checks, each starting with an empty work root."""

import json
import subprocess
import sys

import numpy as np
import pytest

from sosfuse2.atmosphere import density
from sosfuse2.foundation import paths
from sosfuse2.jobs import walkthrough

pytestmark = pytest.mark.offline_cpu


@pytest.fixture
def synthetic_minor_work(tmp_path, monkeypatch):
    """Replace the suite's shared O4 fixture and IPRT cache with an empty offline root."""
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')


def test_reduced_numpy_table(tmp_path):
    # The walkthrough configures process-wide solver and thread settings. Keep them in a fresh interpreter.
    script = """
import json
from pathlib import Path
from sosfuse2.jobs.walkthrough import run_example

record = run_example(Path.cwd())
Path('solved.json').write_text(json.dumps(record['solved']))
"""
    subprocess.run(
        [sys.executable, '-B', '-c', script], cwd=tmp_path,
        env=paths.child_env(walkthrough.example_environment(tmp_path)), check=True, timeout=600,
    )
    solved = json.loads((tmp_path / 'solved.json').read_text())
    assert solved['converged'] == [True]
    assert solved['source_shape'] == [40, 163, 1]
    assert 0 <= solved['observer_change'] <= solved['tolerance'] == 1e-3
    assert 0 <= solved['source_change'] <= solved['tolerance']

    with np.load(tmp_path / 'table.npz') as output:
        np.testing.assert_array_equal(output['wavelength_nm'], [550])
        np.testing.assert_array_equal(output['solar_elevation_deg'], [-3, -6])
        np.testing.assert_array_equal(output['view_elevation_deg'], [78.75, 56.25, 33.75, 11.25])
        np.testing.assert_array_equal(output['relative_azimuth_deg'], [15, 45, 75, 105, 135, 165])
        table = output['radiance']
        assert table.shape == (2, 4, 6, 1)
        assert np.isfinite(table).all()
        # Frozen exp8/triangle/keep radiances at the highest and lowest views, towards and away from the sun.
        expected = [
            [[3.054727e-4, 2.808648e-4], [3.163053e-3, 1.111142e-3]],
            [[1.492353e-5, 1.283569e-5], [2.165351e-4, 4.395965e-5]],
        ]
        np.testing.assert_allclose(table[:, ::3, ::5, 0], expected, rtol=1e-3, atol=0)


def test_default_density_needs_iprt():
    state = walkthrough.example_state()
    del state['density']
    with pytest.raises(RuntimeError, match=r'^E_IPRT_OFFLINE: '):
        density.build(state)
