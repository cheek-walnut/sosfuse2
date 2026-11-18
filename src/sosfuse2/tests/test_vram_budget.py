"""Byte accounting for KSPLIT preflight and persistent admission."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.backends import vram_budget as vram
from sosfuse2.foundation.host_limits import ksplit_fallback_warning
from sosfuse2.solve.ksplit import KsplitMixin


@pytest.mark.parametrize('mode', [0, 1, 2, 3])
def test_decision_items(mode, monkeypatch):
    monkeypatch.delenv('SOSFUSE2_KSPLIT_AERO_MB', raising=False)
    k = KsplitMixin()
    k.K, k.ksplit_mode, k.ksplit_verify = 153, mode, False
    k.host = np.asarray
    k.config = lambda: SimpleNamespace(c3_fw15=True)
    k.ksplit_memory = lambda: 2 << 30
    k.ksplit_other_bytes = lambda: 1 << 30
    state = SimpleNamespace(W=15, geo=SimpleNamespace(h=range(2), chi=range(3)))
    offsets = dict(cell_off=np.array([0, 1, 2]), edge_off=np.array([0, 4, 6]))
    bundles = [(SimpleNamespace(nq=4, cells=2, arrays=offsets), phase) for phase in (1, 0)]
    decision = k.ksplit_preflight(state, bundles)
    assert decision is state.ksplit_decision
    items = json.loads(json.dumps(decision))['items']
    if mode == 0:
        assert items == dict(peak={}, reserve={})
        return
    assert decision['effective'] == mode
    assert sum(items['peak'].values()) == decision['peak']
    assert sum(items['reserve'].values()) + decision['budget'] == decision['solo_free'] == 3 << 30
    assert items['peak']['column_records'] == 2 * (16 + 6 * 16) * 4
    if mode >= 2:
        assert sum(items['planned_peak'].values()) == decision['planned_peak'] == k.ksplit_planned_bytes
        assert items['planned_peak']['vprod_store'] == 2 * 16 * 28


def test_state_peak_uses_transients_from_only_the_largest_stage():
    fixed = vram.bundle_fixed_items(100, 2, 0, 2, 6, 153, 15)
    ground = vram.bundle_fixed_items(80, 1, 1, 2, 6, 153, 15)
    build = vram.vprod_option(fixed.total(), 40, 40, 1, 15, 153, 2, True, 'device', False, 0)
    run = vram.vprod_option(ground.total(), 50, 30, 1, 15, 153, 3, False, 'host', True, 64 << 20)
    assert build['peak'] > build['run'] and run['peak'] == run['run']
    items = vram.state_peak_items([dict(fixed=fixed), dict(fixed=ground)], [build, run])
    assert items['column_records'] == 180
    assert items['vprod_records'] == 8 * (40 + 30) and items['vprod_store'] == 28 * 40
    assert items['plan_workspace'] == run['workspace'] and items['moments_and_split_scratch'] == 0
    assert items['ray_radiances'] == 2 * 4 * 15 * 4
    assert items.total() == run['peak'] + 100 + 8 * 40 + 28 * 40


def test_warning_ranks_peak_and_reserve():
    items = dict(peak=dict(aerosol_chunk=80, column_records=120), reserve=dict(source_fields=100, headroom=20))
    warning = ksplit_fallback_warning(dict(requested=3, effective=0, peak=200, budget=190, items=items))
    assert warning.endswith('largest: peak.column_records=120 B, reserve.source_fields=100 B')


def test_admission_sums(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_FIELD_RESERVE', '1.5')
    state = SimpleNamespace(W=15, strato={}, geo=SimpleNamespace(ns=3))
    fields = int(1.5 * 3 * 1132 * 15 * 4)
    for split, floor in [(False, 0), (True, 5_100_000_000)]:
        admission = vram.admission_items(state, 150, split)
        assert admission == dict(payload_arrays=150, source_fields=fields, ksplit_floor=floor)
        assert admission.total() == 150 + fields + floor
