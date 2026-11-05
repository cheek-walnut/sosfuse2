"""table_q codec (R3 lossy store, sky-sos docs/SPEC_R3_1000state_bake.md section 7 B2): bound, byte planes, explicit failures."""
import numpy as np
import pytest
from .table_q import COORDS, MAX_REL, MAX_WIDTH, TableQError, decode, encode, load, round_trip, save

SHAPE = (3, 4, 5, 2)


def table(path, radiance, **override):
    coords = dict(wavelength_nm=np.array([400., 500.], 'f4'), solar_elevation_deg=np.array([1., 0., -1.], 'f4'),
                  view_elevation_deg=np.linspace(-1, 1, 4).astype('f4'), relative_azimuth_deg=np.linspace(0, 180, 5).astype('f4'),
                  units=np.array('W m-2 sr-1 nm-1'))
    coords.update(override)
    np.savez(path, radiance=np.asarray(radiance, 'f4'), **coords)
    return path


def test_round_trip_within_bound_at_the_width_limit(tmp_path):
    rng = np.random.default_rng(5)
    w = MAX_WIDTH-1e-3                                    # float32 storage moves the ends by ~1e-7 in ln
    r = np.exp(rng.uniform(-w, 0, SHAPE)); r[:, 0, 0, 0] = 1.; r[:, 0, 0, 1] = np.exp(-w)   # every slice at the limit
    f = table(tmp_path/'t.npz', r); save(f, tmp_path/'q.npz')
    row = round_trip(f, tmp_path/'q.npz')
    assert row['max_rel'] <= MAX_REL and row['width'] <= MAX_WIDTH
    back = load(tmp_path/'q.npz')
    assert back['radiance'].dtype == np.float32 and back['radiance'].shape == SHAPE
    assert all(np.array_equal(back[k], np.load(f)[k]) for k in COORDS)


def test_extremes_are_exact_up_to_float32_and_planes_are_high_then_low():
    r = np.array([[1e-10, 3e-6, 0.5], [1e-3, 1e-2, 1.]])  # 2-D: slices are rows, delta axis 0 runs across them
    lo, hi, planes = encode(r)
    d = ((planes[0].astype(np.uint16) << 8) | planes[1]).reshape(r.shape)
    q = np.cumsum(d, axis=0, dtype=np.uint16)
    assert planes.shape == (2, 6) and q[:, 0].tolist() == [0, 0] and q[:, -1].tolist() == [65535, 65535]
    assert np.allclose(lo, np.log(r[:, 0])) and np.allclose(hi, np.log(r[:, -1]))
    back = decode(lo, hi, planes, r.shape)
    assert np.allclose(back[:, [0, -1]], r[:, [0, -1]], rtol=1e-12, atol=0)


def test_delta_wraps_mod_65536_without_loss():
    r = np.exp(np.array([[0., -26., -13.], [-26., 0., 0.], [-13., -13., -26.]]).T)   # +-65535 steps along delta axis 0
    lo, hi, planes = encode(r)
    assert np.allclose(decode(lo, hi, planes, r.shape), r, rtol=MAX_REL, atol=0)


@pytest.mark.parametrize('bad, code', [(0., 'NONPOSITIVE'), (-1e-9, 'NONPOSITIVE'), (np.nan, 'NONFINITE'), (np.inf, 'NONFINITE')])
def test_invalid_values_fail_explicitly(bad, code):
    r = np.full(SHAPE, .1); r[1, 2, 3, 0] = bad
    with pytest.raises(TableQError, match='E_TABLE_Q_'+code):
        encode(r)


def test_constant_and_too_wide_tables_fail():
    r = np.exp(np.linspace(-5, 0, np.prod(SHAPE))).reshape(SHAPE); r[1] = .2
    with pytest.raises(TableQError, match='E_TABLE_Q_CONSTANT: slice 1'):
        encode(r)
    r = np.exp(np.linspace(-5, 0, np.prod(SHAPE))).reshape(SHAPE); r[2, 0, 0, 0] = np.exp(-MAX_WIDTH-0.01)
    with pytest.raises(TableQError, match='E_TABLE_Q_RANGE'):
        encode(r)


def test_round_trip_detects_coordinate_and_member_drift(tmp_path):
    r = np.linspace(.01, 1, np.prod(SHAPE)).reshape(SHAPE)
    save(table(tmp_path/'a.npz', r), tmp_path/'q.npz')
    other = table(tmp_path/'b.npz', r, view_elevation_deg=np.linspace(-2, 2, 4).astype('f4'))
    with pytest.raises(TableQError, match='E_TABLE_Q_COORDS'):
        round_trip(other, tmp_path/'q.npz')
    np.savez(tmp_path/'c.npz', radiance=r.astype('f4'), extra=np.zeros(1), **{k: np.load(tmp_path/'a.npz')[k] for k in COORDS})
    with pytest.raises(TableQError, match='E_TABLE_Q_MEMBERS'):
        save(tmp_path/'c.npz', tmp_path/'q2.npz')


def test_round_trip_fails_beyond_the_bound(tmp_path, monkeypatch):
    from . import table_q
    r = np.exp(np.linspace(-20, 0, np.prod(SHAPE))).reshape(SHAPE)
    f = table(tmp_path/'t.npz', r)
    monkeypatch.setattr(table_q, 'LEVELS', 255)          # coarse quantizer: ~4 % steps
    save(f, tmp_path/'q.npz')
    with pytest.raises(TableQError, match='E_TABLE_Q_ROUNDTRIP'):
        round_trip(f, tmp_path/'q.npz')


def test_reader_rejects_a_damaged_scalar_or_header(tmp_path):
    r = np.exp(np.linspace(-20, 0, np.prod(SHAPE))).reshape(SHAPE)
    save(table(tmp_path/'t.npz', r), tmp_path/'q.npz')
    good = dict(np.load(tmp_path/'q.npz'))
    for key, value, code in (('log_lo', np.where(np.arange(3) == 1, np.nan, good['log_lo']), 'log range'),
                             ('log_hi', good['log_hi']+0.5, 'header vs arrays'),
                             ('shape', np.array([5, 4, 3, 2]), 'E_TABLE_Q_HEADER'), ('q_planes', good['q_planes'][:, :-1], 'planes')):
        np.savez(tmp_path/'bad.npz', **dict(good, **{key: value}))
        with pytest.raises(TableQError, match=code):
            load(tmp_path/'bad.npz')
