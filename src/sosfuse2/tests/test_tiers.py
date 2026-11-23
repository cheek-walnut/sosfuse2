"""The `heavy` marker: the file list in tests/conftest.py names existing test files and the marker reaches their tests."""

from pathlib import Path

from sosfuse2.tests import conftest


def test_heavy_files_exist():
    here = Path(__file__).resolve().parent
    missing = sorted(name for name in conftest.HEAVY_FILES if not (here / name).is_file())
    assert not missing, f'remove from HEAVY_FILES: {missing}'


def test_this_file_is_not_heavy(request):
    assert Path(__file__).name not in conftest.HEAVY_FILES
    assert request.node.get_closest_marker('heavy') is None
