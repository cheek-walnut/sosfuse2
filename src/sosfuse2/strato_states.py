#!/usr/bin/env python3
"""Nine v2 sulfate diagnostic states; only write a supplied scratch output."""
import argparse,copy,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
from sosfuse2.geometry import normal_state
from sosfuse2 import paths
TABLE='assets/strato_sulfate_rg0.08_sg1.6_n1.43.json'   # package-relative (stratosphere.table_path resolves inside the package)

def build():
    states=[];base=normal_state()
    for aod,tau in [(a,t) for a in (.02,.15) for t in (0.,.003,.01,.03)]+[(.15,.10)]:
        s=copy.deepcopy(base);s['id']=f'strato_a{round(aod*1000):03d}_t{round(tau*1000):03d}'
        s['parameters'].update(aod550=aod,stratosphere_tau550=tau)
        s.update(group='strato_purple_stress' if tau==.10 else 'strato_purple',runnable=True,
                 description='local scalar Mie sulfate diagnostic; rig review gate pending',
                 stratosphere=dict(status='enabled',profile='gaussian',center_km=21.,sigma_km=3.,optics_table=TABLE,
                                   polarization='fully depolarizing',phase='Mie scalar L32'))
        states.append(s)
    return dict(version=2,sampling='deterministic Pro purple diagnostic matrix',
                supported_physics='Rayleigh coherency + fully depolarizing trop HG / strat scalar Mie + ozone + Lambert',
                review_gate='astra + Opus review and CUDA/full-grid checks before rig',states=states)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    # Avoid accidental writes outside the agreed module or to the live rig.
    root=paths.work_root()
    if not a.out.resolve().is_relative_to(root):p.error(f'output must be under the work root {root}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(build(),indent=2,allow_nan=False)+'\n');print('S4 9 states:',a.out)
