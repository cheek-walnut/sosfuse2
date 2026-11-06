"""Public CI runs with a synthetic O4 fixture in a private temporary work root."""

import shutil

import pytest


@pytest.fixture(scope='session', autouse=True)
def synthetic_minor_work(tmp_path_factory):
    from sosfuse2 import paths, testing

    original = paths.work_root()
    root = tmp_path_factory.mktemp('synthetic-optics')
    if (original / 'iprt').is_dir():
        shutil.copytree(original / 'iprt', root / 'iprt')
    testing.synthetic_o4(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('SOSFUSE2_WORK_ROOT', str(root))
        patch.setenv('SOSFUSE2_OFFLINE', '1')
        yield root
