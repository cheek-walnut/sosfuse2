"""plan_queue.py against the keys the calculator actually used, and pipeline.py --plan.  CPU only, no CUDA.

Ground truth = PHASE events the calculator printed while it ran full-grid queues:
  rig 2026-10-03: states, run logs and column cache files
  eda 2026-10-07: states and run logs
The packed geometry cache (git-ignored .bin files) and the rig logs live in the
main checkout; from a worktree the test looks there (or in SADDLE_DATA_ROOT) and skips if absent.

Run: python -B -m sosfuse2.tests.test_plan_queue
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from sosfuse2 import paths, pipeline, plan_queue, testing


# r11 generation (2026-10-15, stratosphere.py NOPREFIX port), retained: historical key -> r11 source-hash key.
COLUMN_KEY_R11 = {
    '0b87806d026a4452932b3090886f2fe573a60c29335e638716ae9db4edaade7d': '08c0f2f8ca51e2187285b1e1aa5fb4b3aaae884e1d1ada337436ae056c0dd5c5',
    '142edbdacf805123bfff331038d2d7d57e70768d7aa1bb2f5112c22f8ab0663e': '188a7b39d8d95b47b0a23f2168c7f546624a22ad6fd3807d8b75c499a1adec81',
    '4cbb1fa3fc496f714779ff3da86496a1169e68e3ff152fa0a8590758c349c717': '199e7041fe125e8ec8d6d50ac92936386fa814fab56546a1518ffa4f55f69284',
    '11d2ea963319dd73b102f68dbb4fb1f672433718b3d63e4c188c2f1f118a995d': '3758121abc5ab4e9c545f6a58c41703e8fabd5ca29d487d62e36c83877a3d591',
    'beaef6d7348325485812b2c03766680af224a989366d6dceb9df9ca1d9c94020': 'cd1bf4e923cd6b4cf4235e92d49e9c913ae5221d1a24d9889a3d5fdb5c60525f',
    'b2bdf15dce7ed85f2aad75aebf52f353cf65299e3201494bf71620eb4e7d35f8': 'c00745c0664fec3fe36d0d87c5645bd24c9a624f71b391b37e87504415b52bf8',
}

# r17 generation (2026-10-17, stratosphere.py COMPONENT_RADIUS branch), retained: historical key -> r17 source-hash key.
COLUMN_KEY_R17 = {
    '0b87806d026a4452932b3090886f2fe573a60c29335e638716ae9db4edaade7d': 'f3d90e2803021bf58616e0ee76e81342b0b238d9045648c7ed003b595929e17c',
    '142edbdacf805123bfff331038d2d7d57e70768d7aa1bb2f5112c22f8ab0663e': 'ba2b97c679554c7fc8bb32a91b5fc9ea9c40c71eef824d8c4016bdd647307d14',
    '4cbb1fa3fc496f714779ff3da86496a1169e68e3ff152fa0a8590758c349c717': 'c89f61556f405dacedcb336c92d38888d998d7a60b2fa6f699efe00853720f27',
    '11d2ea963319dd73b102f68dbb4fb1f672433718b3d63e4c188c2f1f118a995d': 'a53074ac23d46f49f9b9e377b085dfcf1fb425b42cae0d08dcc728d59948b47c',
    'beaef6d7348325485812b2c03766680af224a989366d6dceb9df9ca1d9c94020': 'fd378f02fac49a25e451b56ac670c9878b56c9d453d0163cd8ca9ff8a3824980',
    'b2bdf15dce7ed85f2aad75aebf52f353cf65299e3201494bf71620eb4e7d35f8': 'e1c3b3320dd3db873005ef6168d0f11161e31057fc7a5e8501f9d4131d83e069',
}

# Editing the file invalidates the cache on purpose: column_cache.cache_key hashes the whole stratosphere.py file (sources['0']), so
# ff8c05b (2026-10-21: SH order L, attach source_stride) and c6be075 (ground cells via ground_parallel) moved every column
# key.  Neither commit edits columns(); only the source-hash generation changed (stratosphere.py of ff8c05b^ at HEAD
# reproduces the r17 keys).  Historical full keys are retained on the left; current source-hash keys on the right.
COLUMN_KEY_UPDATES = {
    '0b87806d026a4452932b3090886f2fe573a60c29335e638716ae9db4edaade7d': '4d4c67b36f8fccb5ed2eee7a5385aa8e9cb2f42ddb3ea04424cbbb8db1294b91',
    '142edbdacf805123bfff331038d2d7d57e70768d7aa1bb2f5112c22f8ab0663e': 'ae7b42a663f9b9bdf5a30f45fa6ec59f2a4dbe64c3db75eae05c787aba34ad5e',
    '4cbb1fa3fc496f714779ff3da86496a1169e68e3ff152fa0a8590758c349c717': '627ade302b54598075b3a4464492fba2cd2ca2ce3f73a402f4c4d1729931eca1',
    '11d2ea963319dd73b102f68dbb4fb1f672433718b3d63e4c188c2f1f118a995d': 'e4ba847421232453e5fae48b6f30f3b908a45efb1d389ad7b35029ed34dbe3f7',
    'beaef6d7348325485812b2c03766680af224a989366d6dceb9df9ca1d9c94020': 'ff44247a77f1115595bb832fe66249a72e9656b92b85952379f1d6f2f7ccc553',
    'b2bdf15dce7ed85f2aad75aebf52f353cf65299e3201494bf71620eb4e7d35f8': '62858e595ed75be0f77adf04d04e43940e3e0717f390b71d049e15de213df9e2',
}

# The logged runs used nr20/nphi64; production now defaults to V14/nphi16, so replay both explicitly
# (same packed geometry endpoints_bab3478c3f15b4d62418). The engine is unchanged from main.

SIGNATURE = 'endpoints_bab3478c3f15b4d62418'
LUT2 = paths.FIXTURES / 'states_lutv2_strato_usstd.json'
_SKY = testing.sky_sos_root()
PURPLE = _SKY / 'results/purple_aod_axis_20261007/states/states_night.json' if _SKY is not None else None


def data_root():
    """sky-sos research data (packed geometry cache + rig logs): SADDLE_DATA_ROOT, else the SOSFUSE2_SKY_SOS_ROOT checkout
    (and, from a worktree, its main checkout)."""
    candidates = [Path(os.environ['SADDLE_DATA_ROOT'])] if os.environ.get('SADDLE_DATA_ROOT') else []
    if _SKY is not None:
        candidates += [_SKY] + ([_SKY.parents[1]] if _SKY.parent.name == '.worktrees' else [])
    for c in candidates:
        if (c / 'results/sos_fuse/geometry_cache' / SIGNATURE / 'packed/dirs.bin').is_file() and (
            c / 'results/strato_rig_20261003_lut2/runs'
        ).is_dir():
            return c
    return None


DATA = data_root()


def logged(paths):
    """state id -> {'main','ground': key prefix, 'lock': attach lock} from the calculator's PHASE events."""
    out = {}
    for path in paths:
        with path.open(errors='replace') as f:
            lines = f.readlines()
        for line in lines:
            i = line.find('{"event"')
            if i < 0:
                continue
            try:
                e = json.loads(line[i:])
            except ValueError:
                continue
            row = out.setdefault(e.get('id'), {})
            if e.get('phase') == 'column_cache' and e.get('kind') == 'strato' and e['rays'] in (9586008, 404496):
                row.setdefault('main' if e['rays'] == 9586008 else 'ground', set()).add(e['key'])
            if e.get('phase') == 'attach_cache' and e.get('lock'):
                row.setdefault('lock', set()).add(e['lock'])
    return {k: v for k, v in out.items() if v}


def run_plan(out, states, **extra):
    a = plan_queue.parser().parse_args(
        ['--states', str(states), '--geometry-cache', str(DATA / 'results/sos_fuse/geometry_cache'), '--out', str(out)]
        + sum((['--' + k.replace('_', '-'), str(v)] for k, v in extra.items()), [])
    )
    return plan_queue.plan_queue(a)


@unittest.skipIf(DATA is None, 'packed full-grid geometry cache and rig logs not found (set SADDLE_DATA_ROOT)')
class PlanQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = testing.scratch('plan-tests')
        phi64 = cls.root / 'nphi64.json'
        phi64.write_text(json.dumps(dict(nr=20, h_refine="none", nphi=64)))
        cls.lut2 = run_plan(
            cls.root / 'lut2.json',
            LUT2,
            cache_root=DATA / 'results/strato_rig_20261003_lut2/caches',
            cost_profile='rig4090',
            settings_json=phi64,
        )
        cls.purple = run_plan(
            cls.root / 'purple.json',
            PURPLE,
            column_cache=cls.root / 'empty_col',
            attach_cache=cls.root / 'empty_att',
            settings_json=phi64,
        )
        cls.cold = run_plan(
            cls.root / 'cold.json',
            LUT2,
            column_cache=cls.root / 'empty_col',
            attach_cache=cls.root / 'empty_att',
            gpus=2,
            workers_per_gpu=2,
            column_workers=2,
            cost_profile='rig4090',
            settings_json=phi64,
        )
        cls.strato_on = run_plan(
            cls.root / 'strato_on.json',
            PURPLE,
            column_cache=cls.root / 'empty_col',
            attach_cache=cls.root / 'empty_att',
            stratosphere='on',
            settings_json=phi64,
        )

    def keys(self, plan, state_id):
        row = next(r for r in plan['states'] if r['id'] == state_id)
        items = {
            plan['items'][i]['bundle']: plan['items'][i]['key']
            for i in row['items']
            if plan['items'][i]['kind'] == 'column'
        }
        return row, items

    def test_keys_and_groups_match_calculator(self):
        checks = 0
        for plan, logs in (
            (self.lut2, sorted((DATA / 'results/strato_rig_20261003_lut2/runs').glob('*/logs/*.log'))),
            (self.purple, sorted((DATA / 'results/purple_aod_axis_20261007/logs/runs').glob('*/run.raw.log'))),
        ):
            truth = logged(logs)
            ids = {r['id'] for r in plan['states']}
            self.assertTrue(truth and set(truth) <= ids | {'COLUMN_CACHE'})
            for state_id, seen in truth.items():
                row, keys = self.keys(plan, state_id)
                for bundle in ('main', 'ground'):
                    if bundle in seen:
                        expected = {old[:16]: new[:16] for old, new in COLUMN_KEY_UPDATES.items()}
                        self.assertEqual({expected[k] for k in seen[bundle]}, {keys[bundle][:16]}, (state_id, bundle))
                        checks += 1
                if 'lock' in seen:
                    self.assertEqual(seen['lock'], {row['attach_lock']}, state_id)
                    checks += 1
        self.assertGreaterEqual(checks, 40)
        # Existing historical files remain intact; the new source generation is initially cold.
        columns = [it for it in self.lut2['items'].values() if it['kind'] == 'column']
        self.assertEqual(len(columns), 4)
        old_by_new = {new: old for old, new in COLUMN_KEY_UPDATES.items()}
        for it in columns:
            self.assertFalse(it['hit'])
            self.assertEqual(Path(it['file']).name, it['key'] + '.npy')
            self.assertTrue(Path(it['file']).with_name(old_by_new[it['key']] + '.npy').is_file())
        # Groups: one column group and one attach lock per stratosphere profile; stratosphere-off states in none.
        groups = {grp['label']: set(grp['states']) for grp in self.lut2['groups']['column']}
        rows = {r['id']: r for r in self.lut2['states']}
        self.assertEqual(
            groups,
            {
                'center 21 sigma 3': {i for i in rows if i.startswith('dual_')},
                'center 22 sigma 5': {i for i in rows if i.startswith('B22s5_')},
            },
        )
        self.assertEqual(
            {frozenset(g['states']) for g in self.lut2['groups']['attach']}, set(map(frozenset, groups.values()))
        )
        self.assertTrue(
            all(
                not any(self.lut2['items'][i]['kind'] in ('column', 'attach') for i in rows[i]['items'])
                for i in rows
                if i.startswith('off_')
            )
        )
        self.assertEqual(
            [set(g['states']) for g in self.purple['groups']['column']],
            [{r['id'] for r in self.purple['states'] if r['id'].startswith('L19s3_')}],
        )

    def test_order_and_pipeline_plan(self):
        plan = self.cold
        rows = {r['id']: r for r in plan['states']}
        reps = [r for r in plan['states'] if r['role'] == 'representative']
        column_owners = {it['owner'] for it in plan['items'].values() if it['kind'] == 'column'}
        self.assertEqual(len(column_owners), 2)  # one per profile
        self.assertTrue(column_owners <= {r['id'] for r in reps})
        firsts = [g['ids'][0] for g in plan['gpus']]
        self.assertEqual(set(firsts), column_owners)  # spread: one representative leads each GPU
        for g in plan['gpus']:
            roles = [rows[i]['role'] for i in g['ids']]
            self.assertEqual(roles, sorted(roles, key=['representative', 'warm', 'member'].index))
            for i in g['ids']:
                self.assertTrue(set(rows[i]['waits_for']) <= {r['id'] for r in reps})
        self.assertEqual(sorted(i for g in plan['gpus'] for i in g['ids']), sorted(rows))
        # Commands carry the planned states per GPU and column workers; the cold wall divides the column items.
        self.assertEqual((plan['resources']['workers'], plan['resources']['column_workers']), (2, 2))
        self.assertTrue(all(' --workers 2 ' in g['command'] and ' --threads 2 ' in g['command'] for g in plan['gpus']))
        self.assertTrue(
            all(
                r['cold_seconds'] < r['cold_cpu_seconds']
                for r in reps
                if any(
                    plan['items'][i]['kind'] == 'column' and plan['items'][i]['owner'] == r['id'] for i in r['items']
                )
            )
        )
        # pipeline.py --plan resolves one GPU list in order.
        path = self.root / 'cold.json'
        resolve = lambda *extra, plan_path=path, states=LUT2: pipeline.resolve_plan(
            pipeline.parser().parse_args(
                [
                    '--states',
                    str(states),
                    '--plan',
                    str(plan_path),
                    '--settings-json',
                    str(self.root / 'nphi64.json'),
                    *extra,
                ]
            )
        )
        for g in plan['gpus']:
            self.assertEqual(resolve('--plan-gpu', str(g['gpu'])).ids, ','.join(g['ids']))
        # Refused: a changed states file, --ids with --plan, an unknown GPU list.
        changed = self.root / 'changed.json'
        changed.write_text(LUT2.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN_STALE'):
            resolve(states=changed)
        with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN: --ids'):
            resolve('--ids', 'dual_st0p01_aod0p15')
        with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN'):
            resolve('--plan-gpu', '5')
        # Refused: run arguments that change keys or groups (plan with --stratosphere on, run without it; settings).
        on = self.root / 'strato_on.json'
        self.assertTrue(resolve('--stratosphere', 'on', plan_path=on, states=PURPLE).ids)
        with self.assertRaisesRegex(
            ValueError, 'E_PIPELINE_PLAN_STALE: run arguments differ from the plan: stratosphere'
        ):
            resolve(plan_path=on, states=PURPLE)
        settings = self.root / 'settings.json'
        settings.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN_STALE: .*settings_sha256'):
            resolve('--settings-json', str(settings))
        with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN_STALE: .*density_profile'):
            resolve('--density-profile', 'exp8')
        # Refused: GPU lists that repeat or lose a planned state.
        for name, edit in (
            ('duplicate', lambda d: d['gpus'][0]['ids'].append(d['gpus'][1]['ids'][0])),
            ('dropped', lambda d: d['gpus'][1]['ids'].pop()),
        ):
            doc = json.loads(path.read_text())
            edit(doc)
            bad = self.root / (name + '.json')
            bad.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'E_PIPELINE_PLAN: GPU lists'):
                resolve(plan_path=bad)


def test_missing_geometry_cache_is_reported(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from sosfuse2.geometry import normal_state

    state = normal_state()
    state['runnable'] = True
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[state])))
    cache = tmp_path / 'empty_geometry'

    # Historical fixtures above intentionally use nr20/nphi64 cached endpoints. This separate
    # test keeps real empty-cache detection/planning, replacing only expensive optical preparation.
    def probe(state, cfg, geometry_cache, folder):
        assert cfg['nr'] == 44
        geo = SimpleNamespace(
            h=np.arange(cfg['nr'], dtype=float),
            chi=np.linspace(0, np.pi, cfg['nchi']),
            cfg={k: cfg[k] for k in ('nmu', 'nphi', 'path_n')},
            ns=cfg['nr'] * cfg['nchi'],
        )
        s = SimpleNamespace(geo=geo, kernel=SimpleNamespace(height=1.2), fuse_state=state)
        return SimpleNamespace(s=s, scope={}, records=[], old=None)

    monkeypatch.setattr(plan_queue, 'probe', probe)
    args = plan_queue.parser().parse_args(
        ['--states', str(states), '--geometry-cache', str(cache), '--out', str(tmp_path / 'plan.json')]
    )
    plan = plan_queue.plan_queue(args)
    assert len(plan['geometry']) == 1 and plan['geometry'][0]['complete'] is False
    missing = [item for item in plan['items'].values() if item['kind'] == 'geometry']
    assert len(missing) == 1 and missing[0]['hit'] is False
    assert plan['states'][0]['missing'] == ['geometry:' + plan['geometry'][0]['signature']]
    assert not cache.exists()  # Planning reports the work; it never builds the cache.


if __name__ == '__main__':
    unittest.main(verbosity=2)
