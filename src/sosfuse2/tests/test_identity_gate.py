"""Gate calculation identities and reference records, without GPU execution."""

import copy
import json
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2.atmosphere import density
from sosfuse2.checks import gate
from sosfuse2.foundation import content_identity, paths
from sosfuse2.jobs import run_state as v3


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pkg = tmp_path / 'sosfuse2'
    shutil.copytree(paths.GATE, pkg / 'checks/gate_data')
    shutil.copytree(paths.CONFIGS, pkg / 'backends/configs')
    monkeypatch.setattr(gate, 'HERE', pkg)
    monkeypatch.setattr(gate, 'GATE', pkg / 'checks/gate_data')
    monkeypatch.setattr(paths, 'CONFIGS', pkg / 'backends/configs')
    monkeypatch.setattr(gate, 'RECORDS', None)
    monkeypatch.setattr(v3, 'package_hashes', lambda: {'sosfuse2/solve/source_solver.py': 'source'})
    asset = pkg / 'ozone.bin'
    asset.write_bytes(b'profile')
    monkeypatch.setattr(density, 'asset_path', lambda: asset)
    return pkg


@pytest.mark.parametrize('name,keys,value,changed', [
    ('checks/gate_data/fixtures.json', ('note',), 'Updated explanation.', False),
    ('checks/gate_data/fixtures.json', ('settings_note',), 'Updated explanation.', False),
    ('checks/gate_data/fixtures.json', ('state', 'description'), 'Updated explanation.', False),
    ('checks/gate_data/fixtures.json', ('state', 'q2d_override'), 'Updated explanation.', False),
    ('checks/gate_data/fixtures.json', ('settings', 'wavelengths'), 30, True),
    ('checks/gate_data/fixtures.json', ('state', 'parameters', 'aod550'), 0.2, True),
    ('backends/configs/devices.json', ('note',), 'Updated explanation.', False),
    ('backends/configs/devices.json', ('profiles', 'eda', 'launch_budget_seconds'), 20, True),
    ('backends/configs/C3.json', ('notes',), {'definition': 'Updated explanation.'}, False),
    ('backends/configs/C3.json', ('kernel_flags',), ['FMAD'], True),
])
def test_identity_json_changes(inputs, name, keys, value, changed):
    before = gate.code_identity()
    p = inputs / name
    doc = json.loads(p.read_text())
    target = doc
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    p.write_text(json.dumps(doc, indent=4) + '\n')
    assert (gate.code_identity() != before) is changed


def test_identity_indentation(inputs):
    before = gate.code_identity()
    p = inputs / 'checks/gate_data/fixtures.json'
    p.write_text(json.dumps(json.loads(p.read_text()), indent=4))
    assert gate.code_identity() == before


def test_plan_state_notes_and_parameters(inputs, tmp_path):
    F = gate.load_fixtures()
    state = copy.deepcopy(F['state'])
    sid = state['id']
    rad = np.ones((2, 3))
    ref = tmp_path / 'reference.npz'
    np.savez(ref, radiance=rad)
    gate.jdump(gate.GATE / 'baselines/manifest.json', {'gf': {'radiance_sha256': gate.sha(rad)}})
    state.update(description='Updated description.', note_enable='Enabled.', runnable=False)
    gate.plan_check([sid], {sid: state}, {}, ref)
    state['parameters']['aod550'] += 0.1
    with pytest.raises(SystemExit, match='E_GATE_FIRST_STATE: product state definition differs'):
        gate.plan_check([sid], {sid: state}, {}, ref)


def test_reference_required_even_when_manifest_file_exists(inputs):
    ref = gate.GATE / 'fixed.npz'
    np.savez(ref, radiance=np.ones(2))
    gate.jdump(gate.GATE / 'baselines/manifest.json', {
        'gf': {'reference': str(ref), 'radiance_sha256': 'a' * 64},
    })
    with pytest.raises(gate.GateError, match='E_GATE_BASELINE_MISSING') as exc:
        gate.gf(NS(reference=None), gate.load_fixtures(), {})
    assert 'fixed.npz' in str(exc.value) and 'a' * 16 in str(exc.value)
    assert str(ref.parent) not in str(exc.value)


def test_portable_home(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert gate.portable(tmp_path / 'sub/table.npz') == 'table.npz'
    assert gate.portable('~/sub/table.npz') == 'table.npz'


def test_manifest_files_and_paths():
    refs = set()

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('file', 'previous') and isinstance(child, str):
                    refs.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert not value.startswith(('~/', '/'))

    base = paths.GATE / 'baselines'
    visit(json.loads((base / 'manifest.json').read_text()))
    assert refs == {p.relative_to(base).as_posix() for p in base.rglob('*.npz')}


def test_config_prose_is_notes():
    for name in ('C3', 'C3_nostrato', 'devices'):
        doc = json.loads((paths.CONFIGS / f'{name}.json').read_text())
        rows = [doc, *doc.get('profiles', {}).values()]
        for row in rows:
            for key, value in row.items():
                if isinstance(value, str):
                    assert content_identity.is_note(key) or key == 'flagset'
                if isinstance(value, list) and any(isinstance(v, str) for v in value):
                    assert key in ('kernel_flags', 'gpu_names', 'stage1_defaults') or content_identity.is_note(key)


def test_stage_toolkey_fields_and_old_staged_rejection(tmp_path, monkeypatch):
    F = gate.load_fixtures()
    meta = dict(device=dict(cc='8.6', nvrtc='12', profile='default'),
                config=dict(flags=[], stage1=[], async_=0, L=16, block=256),
                launch=dict(max_seconds=0.0))
    row = dict(meta=meta)
    monkeypatch.setattr(gate, 'run_all', lambda *a, **k: {'FX-C3': row})
    monkeypatch.setattr(gate, 'repeat_run', lambda *a: dict(pass_=True, repeat_max_rel=0.0))
    monkeypatch.setattr(gate, 'outputs', lambda r: {'radiance': 'test'})
    args = NS(flagsets=None, fixtures='FX-C3', inject=None, bless_numpy=False, bless=False,
              stage=True, runs=2, out=tmp_path, no_historical=True, no_persistent=True)
    report = dict(gates={})
    assert gate.dispatch(args, report)
    result = report['fixtures']['FX-C3']
    key, fields = gate.toolkey(meta)
    assert result['toolkey'] == key and isinstance(result['toolkey'], str)
    assert result['toolkey_fields'] == fields
    monkeypatch.setattr(gate, 'code_identity', lambda: 'current')
    for old in (None, 'raw-byte-identity'):
        meta['code_identity'] = old
        with pytest.raises(gate.GateError, match='E_GATE_CONFIG: .*code_identity.*stage again'):
            gate.bless(F, ['FX-C3'], NS(staged=tmp_path, out=tmp_path), {})


def test_old_staged_error_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'run_all', lambda *a, **kw: {'FX-C3': {'meta': {'code_identity': 'old'}}})
    monkeypatch.setattr(gate, 'code_identity', lambda: 'current')
    monkeypatch.setattr(gate, 'baselines_writable', lambda: True)
    rc = gate.main(['--bless', '--staged', str(tmp_path / 'staged'), '--out', str(tmp_path / 'out'),
                    '--fixtures', 'FX-C3'])
    report = json.loads((tmp_path / 'out/gate.json').read_text())
    assert rc == 2 and report['code'] == 'E_GATE_CONFIG'
    assert 'code_identity' in report['error'] and 'stage again' in report['error']


def test_density_asset_keeps_raw_hash(inputs):
    before = gate.code_identity()
    (inputs / 'ozone.bin').write_bytes(b'profile\n')
    assert gate.code_identity() != before
