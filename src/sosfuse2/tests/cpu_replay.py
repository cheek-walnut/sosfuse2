"""CPU replay of one FW=15 chunk of a gate fixture (test support; pytest does not collect this module).

replay() runs the chunk loop of gate.worker for a single chunk of the 45-wavelength fixtures on the NumPy backend: the
same public calls (settings, state, create_configured, Solver.apply, query), a few applies instead of all of them. The
result is compared with the matching slice of the recorded NumPy golden (golden_slice) or, for a fixture without one,
of the canonical CUDA record (canonical_slice).

Limits:
  - The NumPy path does not execute kernel.cu. A change made only there is invisible here; it needs the CUDA gate.
  - Only chunk-aligned slices (bands 0-14, 15-29, 30-44) are comparable with a golden: the kernel is specialised to
    15 wavelengths and the recorded arrays were produced chunk by chunk.
  - Bands below 665.5 nm depend on the O4 asset of the work root; a synthetic asset does not reproduce them.
  - After an intended physics change the golden comparison is expected to fail. The CLI output is then the
    difference report (per fixture and chunk: maximum relative difference, per solar elevation and per band).

CLI (uses the work root of the caller, with its prepared O4 asset):
  python -B -m sosfuse2.tests.cpu_replay [--fixture FX-C3 FX-N FX-AP] [--chunks 0 1 2] [--applies 2] [--rtol 1e-3]
      [--out DIR]
Exit 0 when every row passes, 1 otherwise. Without the O4 asset the engine stops with E_O4_ASSET_MISSING.
"""

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from sosfuse2 import gate, testing

FW = 15  # wavelengths per kernel chunk


@contextlib.contextmanager
def controls(env):
    """The kernel controls of a fixture in os.environ (and no attach cache), restored on exit, errors included."""
    names = (*gate.CONTROLS, 'STRATO_ATTACH_CACHE')
    saved = {n: os.environ.get(n) for n in names}
    try:
        for n in gate.CONTROLS:
            os.environ[n] = env[n]
        os.environ.pop('STRATO_ATTACH_CACHE', None)  # cold attach, as in the gate worker
        yield
    finally:
        for n, value in saved.items():
            if value is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = value


def band_slice(radiance, chunk, applies):
    """(apply, solar, direction, band, 1) -> the first `applies` applies and the FW bands of `chunk`."""
    return np.asarray(radiance)[:applies][..., FW * chunk : FW * chunk + FW, :]


def replay(fx, chunk, applies, out, cache):
    """Radiance of fixture `fx`, chunk `chunk` (0, 1, 2), after each of `applies` NumPy applies.

    Shape (applies, solar elevation, direction, band, 1) = (applies, 4, 15, 15, 1). `out` receives the settings file and
    the prepare folder, `cache` is the host geometry cache (shareable between calls).
    """
    from sosfuse2 import run_gpu_state_v3 as v3
    from sosfuse2.fused import Solver
    from sosfuse2.geometry import TransientGeometry
    from sosfuse2.stratosphere import source_size

    F = gate.load_fixtures()
    out = Path(out)
    with controls(gate.fixture_env(F, fx)):
        block = int(os.environ['SOSFUSE2_BLOCK_SIZE'])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'settings.json').write_text(json.dumps(gate.fixture_settings(F, fx)))
        cfg = v3.settings_for(NS(reduced=True, settings_json=out / 'settings.json'))
        state = v3.effective_state(gate.fixture_state(F, fx), NS())
        cfg = v3.resolve_settings(cfg, state)
        wl = np.asarray(cfg['wavelengths_nm'])[FW * chunk : FW * chunk + FW]
        if len(wl) != FW:
            raise ValueError(f'E_TEST_CONFIG: chunk {chunk} of {fx} holds {len(wl)} wavelengths, expected {FW}')
        el, az = (np.asarray(F['directions'][n], float) for n in ('elevation_deg', 'azimuth_deg'))
        s, scope = v3.create_configured(out / 'prepare', cfg, wl, 'numpy', state, block, geometry_cache=Path(cache))
        solver = Solver(s, TransientGeometry(s, scope))
        k = s.kernel
        field = k.xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        radiance = []
        for _ in range(applies):
            field = solver.apply(field)
            host = k.host(field)
            full = np.zeros((*host.shape[:2], 2 * s.W), 'f4')
            full[..., ::2] = host
            radiance.append(
                np.stack(
                    [
                        (s.query(h, el, az, full, True) + s.query(h, el, az, full, False))[..., ::2]
                        for h in F['observer_solar_deg']
                    ]
                )
            )
    return np.stack(radiance)


def golden_slice(fx, chunk, applies):
    """The recorded NumPy golden of the fixture (file hash checked), cut like replay()."""
    name = gate.load_fixtures()['fixtures'][fx]['golden']
    return band_slice(gate.golden_set(gate.manifest(), [name])[name]['radiance'], chunk, applies)


def canonical_slice(fx, chunk, applies):
    """The canonical CUDA record of the fixture (file hash checked), cut like replay()."""
    entries = [e for e in gate.manifest()['cuda'].get(fx, []) if e.get('canonical')]
    if len(entries) != 1:
        raise gate.GateError('E_GATE_BASELINE_MISSING', f'{fx}: exactly one canonical CUDA baseline required')
    e = entries[0]
    return band_slice(gate.baseline_npz(e['file'], e['sha256'])['fixed_radiance'], chunk, applies)


def compare(actual, ref, rtol):
    """|actual - ref| <= rtol * |ref| in float64, no absolute allowance; arrays shaped like replay()."""
    a, r = np.asarray(actual, 'f8'), np.asarray(ref, 'f8')
    if a.shape != r.shape:
        return dict(
            pass_=False, reason=f'shape {a.shape} vs {r.shape}', max_rel=None, worst=None, band_max_rel=None, s_max=None
        )
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    delta, denom = np.abs(a - r), np.abs(r)
    rel = np.divide(delta, denom, out=np.where(delta == 0, 0.0, np.inf), where=denom != 0)
    rel = np.where(np.isnan(rel), np.inf, rel)  # a non-finite element is the worst one
    return dict(
        pass_=finite and bool(np.all(delta <= rtol * denom)),
        reason=None if finite else 'nonfinite',
        max_rel=float(rel.max(initial=0.0)),
        worst=[int(i) for i in np.unravel_index(int(np.argmax(rel)), rel.shape)] if rel.size else [],
        band_max_rel=[float(x) for x in np.moveaxis(rel, -2, 0).reshape(rel.shape[-2], -1).max(axis=1, initial=0.0)],
        s_max=gate.elevation_stats(a[-1], r[-1], gate.load_fixtures()['observer_solar_deg'])['S_max'],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fixture', nargs='+', default=['FX-C3', 'FX-N', 'FX-AP'])
    p.add_argument('--chunks', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--applies', type=int, default=2)
    p.add_argument('--rtol', type=float, default=gate.RTOL)
    p.add_argument('--out', type=Path, help='work folder outside the package (default: a new scratch folder)')
    a = p.parse_args(argv)
    out = testing.outside_package(a.out) if a.out else Path(tempfile.mkdtemp(dir=testing.scratch_parent()))
    fixtures = gate.load_fixtures()['fixtures']
    ok = True
    for fx in a.fixture:
        cut, source = (golden_slice, 'golden') if fixtures[fx]['golden'] else (canonical_slice, 'cuda-canonical')
        for chunk in a.chunks:
            actual = replay(fx, chunk, a.applies, out / f'{fx}-{chunk}', out / 'geometry_cache')
            row = compare(actual, cut(fx, chunk, a.applies), a.rtol)
            ok &= row['pass_']
            head = f"{fx} chunk={chunk} applies={a.applies} ref={source} rtol={a.rtol:g} pass_={row['pass_']}"
            if row['max_rel'] is None:
                print(head, row['reason'], flush=True)
                continue
            print(
                head,
                f"max_rel={row['max_rel']:.3e}",
                'worst=' + json.dumps(row['worst'], separators=(',', ':')),
                's_max=' + json.dumps({h: float(f'{v:.3e}') for h, v in row['s_max'].items()}, separators=(',', ':')),
                'band_max_rel=' + json.dumps([float(f'{v:.3e}') for v in row['band_max_rel']], separators=(',', ':')),
                flush=True,
            )
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
