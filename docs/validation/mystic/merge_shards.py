"""Merge independent shard runs with exact task coverage and no duplicate task IDs."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from common import add_paths, dump, sha, work_path
from run_mystic import FIELDS, build_jobs


def key(job):
    return job['case'], job['wi'], job['seed_id']


def merge(sources, dest, H, config, arms):
    c = json.loads(config.read_text())
    data = json.loads((H / 'out/optics.json').read_text())
    expected = {key(j) for j in build_jobs(c, data, '', '', False, (1, 1))}
    seen = set()
    shards = set()
    rows = []
    counts = dict(planned=0, success=0, failed=0, not_run=0)
    manifests = []
    total_n = None
    plans = []
    common_identity = None
    for source in sources:
        out = source / 'out' if (source / 'out').is_dir() else source
        source_root = out.parent
        m = json.loads((out / 'manifest.json').read_text())
        p = json.loads((out / 'plan_full.json').read_text())
        if m['optics_sha256'] != sha(H / 'out/optics.json') or m['config_sha256'] != sha(config):
            raise ValueError('E_SHARD_INPUT_SHA')
        fields = (
            'uvspec_sha256',
            'libradtran_data_sha256',
            'code_sha256',
            'input_sha256',
            'config_sha256',
            'optics_sha256',
        )
        identities = tuple(m[field] for field in fields)
        photons = {j['photons'] for j in p['jobs']}
        if len(photons) != 1 or photons != {c['photons']}:
            raise ValueError('E_SHARD_PHOTONS')
        if 'photons' in m and m['photons'] != c['photons']:
            raise ValueError('E_SHARD_PHOTONS')
        if common_identity is not None and identities != common_identity:
            raise ValueError('E_SHARD_IDENTITY')
        common_identity = identities
        if m['results_sha256'] != sha(out / 'full.csv'):
            raise ValueError('E_SHARD_CSV_SHA')
        k, n = p['shard']
        if (k, n) in shards or (total_n is not None and n != total_n):
            raise ValueError('E_SHARD_DUPLICATE_OR_COUNT')
        shards.add((k, n))
        total_n = n
        partition = {key(j) for j in build_jobs(c, data, '', '', False, (k, n))}
        actual = {key(j) for j in p['jobs']}
        if actual != partition or len(actual) != len(p['jobs']):
            raise ValueError('E_SHARD_PLAN_COVERAGE')
        if seen & actual:
            raise ValueError('E_DUPLICATE_TASK')
        seen |= actual
        with (out / 'full.csv').open(newline='') as f:
            rr = list(csv.DictReader(f))
        expected_run_ids = {f'{j["case"]}_w{j["wi"]:02d}_s{j["seed_id"]}' for j in p['jobs']}
        if {r['run_id'] for r in rr} != expected_run_ids or len(rr) != len(expected_run_ids):
            raise ValueError('E_SHARD_CSV_COVERAGE')
        for field, status in [('success', 'ok'), ('failed', 'failed'), ('not_run', 'not_run')]:
            observed = sum(r['status'] == status for r in rr)
            if observed != m['counts'][field]:
                raise ValueError('E_SHARD_COUNT')
            counts[field] += observed
        counts['planned'] += len(rr)
        rows += rr
        manifests.append(dict(source_manifest_sha256=sha(out / 'manifest.json'), **m))
        plans.append((source_root, p))
    missing = expected - seen
    if missing or shards != {(k, total_n) for k in range(1, total_n + 1)}:
        raise ValueError('E_MISSING_TASK_OR_SHARD ' + str(len(missing)))
    # Validate before any output mutation; copying is additive and never overwrites existing runs.
    if (dest / 'out/full.csv').exists() or (dest / 'runs').exists():
        raise ValueError('E_MERGE_DEST_EXISTS')
    (dest / 'out').mkdir(parents=True, exist_ok=True)
    for source, p in plans:
        for j in p['jobs']:
            rel = Path('runs') / j['group'] / j['case'] / f'w{j["wi"]:02d}' / f's{j["seed_id"]}'
            src = source / rel
            if src.exists():
                result = src / 'result.json'
                if result.exists() and json.loads(result.read_text())['identity'] != j['identity']:
                    raise ValueError('E_SHARD_RUN_SHA')
                shutil.copytree(src, dest / rel)
    with (dest / 'out/full.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['run_id']))
    for source, name in ((config, 'config.json'), (arms, 'arms.json'), (H / 'states.json', 'states.json')):
        if source.exists() and (dest / name).resolve() != source.resolve():
            shutil.copy2(source, dest / name)
    # The engine reads the parent bundle named by states.json from the same folder.
    parent = json.loads((H / 'states.json').read_text()).get('parent') if (H / 'states.json').exists() else None
    if parent and (dest / parent).resolve() != (H / parent).resolve():
        if not (H / parent).is_file():
            raise ValueError('E_STATES_PARENT: ' + parent + ' is missing next to states.json')
        shutil.copy2(H / parent, dest / parent)
    if (dest / 'out/optics.json').resolve() != (H / 'out/optics.json').resolve():
        shutil.copy2(H / 'out/optics.json', dest / 'out/optics.json')
    dump(
        dest / 'out/manifest.json',
        dict(
            schema_version='v1-merged-mc-1',
            optics_sha256=sha(H / 'out/optics.json'),
            config_sha256=sha(config),
            results_sha256=sha(dest / 'out/full.csv'),
            counts=counts,
            complete=counts['failed'] == counts['not_run'] == 0,
            shard_manifests=manifests,
            coverage=dict(duplicate=0, missing=0),
        ),
    )
    print('MERGE', counts)
    return counts


def main():
    ap = argparse.ArgumentParser()
    add_paths(ap, arms=True)
    ap.add_argument('--shards', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    merge(a.shards, a.output.resolve(), work_path(a.work), a.config, a.arms_file)


if __name__ == '__main__':
    main()
