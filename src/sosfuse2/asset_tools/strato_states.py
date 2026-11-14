#!/usr/bin/env python3
"""Nine v2 sulfate diagnostic states; only write a supplied scratch output."""

import argparse, copy, json, sys
from pathlib import Path

if __name__ == '__main__':
    sys.dont_write_bytecode = True
from sosfuse2.solve.ray_geometry import normal_state
from sosfuse2.foundation import paths

TABLE = 'assets/strato_sulfate_rg0.08_sg1.6_n1.43.json'  # package-relative (stratosphere.table_path resolves inside the package)


def build():
    states = []
    base = normal_state()
    for aod, tau in [(a, t) for a in (0.02, 0.15) for t in (0.0, 0.003, 0.01, 0.03)] + [(0.15, 0.10)]:
        s = copy.deepcopy(base)
        s['id'] = f'strato_a{round(aod * 1000):03d}_t{round(tau * 1000):03d}'
        s['parameters'].update(aod550=aod, stratosphere_tau550=tau)
        s.update(
            group='strato_purple_stress' if tau == 0.10 else 'strato_purple',
            runnable=True,
            description='scalar Mie sulfate diagnostic state',
            stratosphere=dict(
                status='enabled',
                profile='gaussian',
                center_km=21.0,
                sigma_km=3.0,
                optics_table=TABLE,
                polarization='fully depolarizing',
                phase='Mie scalar L32',
            ),
        )
        states.append(s)
    return dict(
        version=2,
        sampling='deterministic stratospheric diagnostic matrix',
        supported_physics='Rayleigh coherency + fully depolarizing trop HG / strat scalar Mie + ozone + Lambert',
        states=states,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    # Keep generated states under the work root.
    root = paths.work_root()
    if not a.out.resolve().is_relative_to(root):
        p.error(f'output must be under the work root {root}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(build(), indent=2, allow_nan=False) + '\n')
    print('S4 9 states:', a.out)
