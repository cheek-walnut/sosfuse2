"""Materialize settings and signed input identities."""

import argparse
from types import SimpleNamespace

from common import add_paths, configure_root, dump, package_identity, read, sha, work_path


def main():
    parser = argparse.ArgumentParser()
    add_paths(parser, arms=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--status')
    parser.add_argument('--reason', default='')
    args = parser.parse_args()
    work = work_path(args.work)
    configure_root(work)
    arms = read(args.arms_file)
    if args.arm not in ('A0', 'A1', 'A2') or args.arm not in arms:
        raise ValueError('E_UNKNOWN_ARM')
    dest = work / 'engine' / args.arm
    if args.status:
        dump(dest / 'progress.json', dict(engine_config_id=args.arm, status=args.status, reason=args.reason))
        return
    from sosfuse2.foundation import paths
    from sosfuse2.jobs import run_state as engine

    c = read(args.config)
    settings = dict(c['settings'], **arms[args.arm])
    dump(dest / 'settings.json', settings)
    cfg = engine.settings_for(SimpleNamespace(reduced=False, settings_json=dest / 'settings.json'))
    if cfg.get('kernel', {}).get('ground_sh', {}).get('order') != 24:
        raise ValueError('E_GROUND_SH')
    if engine.ANDERSON_TOLERANCE != 0.001:
        raise ValueError('E_TOLERANCE')
    identity = package_identity()
    contract = dict(
        engine_config_id=args.arm,
        overrides=arms[args.arm],
        settings=cfg,
        tolerance=0.001,
        engine_version=identity['version'],
        engine_commit=identity['commit'],
        states_sha256=sha(work / 'states.json'),
        states_parent_sha256=sha(paths.ASSETS / 'states_v2.json'),
        optics_sha256=sha(work / 'out/optics.json'),
    )
    path = dest / 'contract.json'
    if path.exists() and read(path) != contract:
        raise ValueError('E_ARM_RESUME_SHA ' + args.arm)
    dump(path, contract)


if __name__ == '__main__':
    main()
