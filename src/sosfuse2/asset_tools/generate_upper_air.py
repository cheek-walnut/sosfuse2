"""Generate a normalized US76 effective Rayleigh number-density profile.

See THIRD_PARTY.md section 8 for the molecular cross sections and atomic polarizabilities.
The middle, 550 nm column is used; the other columns quantify spectral variation.
ussa1976 is a generator dependency only.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

RATIO_TO_AIR = {'N2': (1.0255807727705282, 1.0269982826233435, 1.0273452361327522), 'O2': (0.9076541480897997, 0.9022907037107418, 0.9009674442794132), 'Ar': (0.8812902358555402, 0.8832255568699436, 0.8840244795742395), 'CO2': (2.3250575780018115, 2.320898599920126, 2.3175452780899226), 'O': (0.2016467268339067, 0.2020895447978984, 0.20227234513059258), 'He': (0.013670915862673261, 0.01370093731267774, 0.013713330511452492), 'Ne': (0.0507928650902809, 0.0509044066632969, 0.050950452303525136), 'H': (0.1453665439083877, 0.14568577010172457, 0.14581755033444285), 'H2': (0.20932782322807836, 0.2097875089464834, 0.20997727248159778), 'Kr': (2.026086585318684, 2.030535889062259, 2.0323726126613906), 'Xe': (5.350134889357149, 5.361883832055027, 5.3667339303089845), 'CH4': (2.148481625992166, 2.153199710308402, 2.15514738960965)}
VERSION = '0.3.4'
ASSET = Path(__file__).resolve().parents[1] / 'assets' / 'upper_air_us76_neff.json'


def generate():
    import numpy as np
    try:
        import ussa1976
    except ModuleNotFoundError as exc:
        if exc.name != 'ussa1976':
            raise
        raise RuntimeError(
            f'E_UPPER_AIR_GENERATOR_DEP: python -m pip install ussa1976=={VERSION}'
        ) from exc

    version = importlib.metadata.version('ussa1976')
    if version != VERSION:
        raise RuntimeError(f'E_UPPER_AIR_GENERATOR_VERSION: expected {VERSION}, got {version}')
    z = np.arange(961, dtype='f8') * .125 + 80.
    # compute accepts geometric metres and handles the geopotential conversion internally.
    ds = ussa1976.compute(z=z * 1000.)
    species = [str(s) for s in ds['s'].values]
    n = np.asarray(ds['n'].transpose('s', 'z').values, 'f8')
    total = np.asarray(ds['n_tot'].values, 'f8')
    fractions = n / total[None, :]
    ratios = np.array([RATIO_TO_AIR[s] for s in species], 'f8')
    factors = fractions.T @ ratios
    f = factors[:, 1]
    effective = total * f
    normalized = effective / effective[0]
    return dict(
        version=1, z_km=z.tolist(), neff_ratio=normalized.tolist(),
        metadata=dict(
            source_package='ussa1976', source_version=version, generator='asset_tools/generate_upper_air.py',
            geometric_height=True, step_km=.125, reference_wavelength_nm=550.,
            formula='n_eff(z) = n_tot(z) * sum_i (n_i(z) / n_tot(z)) * sigma_i / sigma_air',
            normalization='n_eff(z) / n_eff(80 km)', n_tot_80_m3=float(total[0]),
            n_eff_80_m3=float(effective[0]), f_80=float(f[0]),
            f_at_km={str(h): float(f[int((h - 80) / .125)]) for h in (100, 120, 200)},
            cross_section_method='Molecular cross sections for N2, O2, Ar and CO2 and static dipole polarizabilities relative to Ar for atoms (THIRD_PARTY.md, section 8)',
            wavelengths_nm=[450., 550., 650.], ratio_to_air=RATIO_TO_AIR,
            factor_spectral_max_rel=float(np.max(np.abs(factors / f[:, None] - 1))),
            species_sum_max_rel=float(np.max(np.abs(fractions.sum(axis=0) - 1))),
            serialization='UTF-8, sorted keys, indent=2, finite JSON numbers, trailing LF',
        ),
    )


def encoded():
    return (json.dumps(generate(), sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f'Requires ussa1976=={VERSION}.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = args.output
    if args.check:
        output = output if output is not None else ASSET
    elif output is None:
        parser.error('--output is required for generation')
    elif output.resolve() == ASSET.resolve() or (output.exists() and ASSET.exists() and output.samefile(ASSET)):
        parser.error('--output must differ from the bundled asset')
    data = encoded()
    digest = hashlib.sha256(data).hexdigest()
    if args.check:
        if not output.is_file() or hashlib.sha256(output.read_bytes()).hexdigest() != digest:
            raise RuntimeError('E_UPPER_AIR_GENERATOR_CHECK: regenerated asset SHA differs')
    else:
        output.write_bytes(data)
    print(json.dumps(dict(status='UPPER_AIR_CHECK_OK' if args.check else 'UPPER_AIR_GENERATED', sha256=digest)))


if __name__ == '__main__':
    main()
