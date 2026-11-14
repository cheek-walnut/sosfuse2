"""Shared paths, input identity and state selection."""

import hashlib
import json
import os
import shutil
import uuid
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'config.json'
DEFAULT_ARMS = HERE / 'arms.json'
DENSITY = {'density_profile': 'usstd76', 'ozone_profile': 'usstd76', 'rayleigh_column': 'usstd'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def work_path(path):
    work = Path(path).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / 'out').mkdir(exist_ok=True)
    return work


def configure_root(work):
    if not os.environ.get('SOSFUSE2_WORK_ROOT', '').strip():
        os.environ['SOSFUSE2_WORK_ROOT'] = str(work / 'sosfuse2_root')


def package_identity():
    dist = metadata.distribution('sosfuse2')
    text = dist.read_text('direct_url.json')
    direct = json.loads(text) if text else {}
    return {'version': dist.version, 'commit': direct.get('vcs_info', {}).get('commit_id')}


def make_states(work, config):
    from sosfuse2.foundation import paths

    source = paths.ASSETS / 'states_v2.json'
    bundle = read(source)
    selected = set(config['state_ids'])
    states = [dict(state, density=dict(DENSITY)) for state in bundle['states'] if state['id'] in selected]
    if {state['id'] for state in states} != selected or len(states) != len(config['state_ids']):
        raise ValueError('E_STATE_MISSING')
    bundle['states'] = states
    dump(work / 'states.json', bundle)
    # The engine also reads the parent file named by the bundle, from the folder of states.json.
    if bundle.get('parent'):
        shutil.copyfile(source.parent / bundle['parent'], work / bundle['parent'])
    return source


def add_paths(parser, *, arms=False):
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    if arms:
        parser.add_argument('--arms-file', type=Path, default=DEFAULT_ARMS)
