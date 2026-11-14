"""Documentation pointers: the files, functions, commands and links that AGENTS.md, docs/ARCHITECTURE.md and the start
of README.md name exist, and docs/ENVIRONMENT.md lists every SOSFUSE2_ variable that the code names.

The documents sit at the root of a source checkout and the installed package does not contain them, so every test skips
from an installed copy (reason given). Pointers are backticked, as `layer/module.py` or `layer/module.py::name`
(`Class.method` is accepted); paths resolve against src/sosfuse2/ first, then against the checkout root.
Run: python -m pytest -q src/sosfuse2/tests/test_docs_pointers.py
"""

import ast
import re

import pytest

from sosfuse2.foundation import paths

ROOT = paths.PKG.parent.parent
POINTED = ('AGENTS.md', 'docs/ARCHITECTURE.md')
LAYERS = (
    'foundation', 'atmosphere', 'rays', 'precompute', 'backends', 'solve', 'outputs', 'jobs', 'checks', 'asset_tools'
)
BARE = ('__main__.py', 'AGENTS.md', 'README.md')
POINTER = re.compile(r'`([A-Za-z0-9_./-]+\.(?:py|cu|cpp|json|md))(?:::([A-Za-z_][A-Za-z0-9_.]*))?`')


@pytest.fixture(scope='module')
def texts():
    names = (*POINTED, 'README.md', 'docs/ENVIRONMENT.md')
    if not (ROOT / 'pyproject.toml').is_file() or not all((ROOT / n).is_file() for n in names):
        pytest.skip('AGENTS.md and docs/ are at the root of a source checkout, not in the installed package')
    return {n: (ROOT / n).read_text(encoding='utf-8') for n in names}


def locate(name):
    return next((p for p in (paths.PKG / name, ROOT / name) if p.exists()), None)


def defined(path, dotted):
    nodes = ast.parse(path.read_text(encoding='utf-8')).body
    for part in dotted.split('.'):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part:
                nodes = node.body
                break
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in node.targets):
                nodes = []
                break
        else:
            return False
    return True


def pointers(text):
    """(path, name or None) of every pointer outside the layer table, whose bare module names are checked on their own.
    A bare file name counts only when it is one of BARE; names such as meta.json are not locations."""
    rows = [ln for ln in text.splitlines() if not re.match(r'\| `(%s)` \|' % '|'.join(LAYERS), ln)]
    return [(n, f) for n, f in POINTER.findall('\n'.join(rows)) if '/' in n or f or n in BARE]


def start_here(readme):
    return readme.split('## Start here', 1)[1].split('\n## ', 1)[0]


def test_pointed_files_and_functions_exist(texts):
    missing = []
    for doc, text in ((n, texts[n]) for n in POINTED):
        for name, func in pointers(text):
            found = locate(name)
            if found is None or (func and not (found.suffix == '.py' and defined(found, func))):
                missing.append(f'{doc}: {name}' + (f'::{func}' if func else ''))
    assert not missing, 'pointers to nothing: ' + '; '.join(missing)


def test_readme_start_here_pointers_exist(texts):
    missing = [name for name, _ in pointers(start_here(texts['README.md'])) if locate(name) is None]
    assert not missing, f'README.md, Start here: {missing}'


def test_layer_table_modules_exist(texts):
    rows = [ln for ln in texts['docs/ARCHITECTURE.md'].splitlines() if re.match(r'\| `(%s)` \|' % '|'.join(LAYERS), ln)]
    assert [r.split('`')[1] for r in rows] == list(LAYERS)
    missing = []
    for row in rows:
        layer = row.split('`')[1]
        for token in re.findall(r'`([^`]+)`', row.split('|')[3]):
            target = paths.PKG / layer / (token if '.' in token or token.endswith('/') else token + '.py')
            if not target.exists():
                missing.append(f'{layer}/{token}')
    assert not missing, f'layer table names modules that are not there: {missing}'


def test_import_exception_table_names_real_functions(texts):
    section = texts['docs/ARCHITECTURE.md'].split('## 6.', 1)[1].split('\n## ', 1)[0]
    rows = [ln.split('|')[1:5] for ln in section.splitlines() if re.match(r'\| `[a-z_/]+\.py` \|', ln)]
    assert len(rows) == 12, 'section 6 lists twelve module pairs'
    bad = []
    for importer, imported, funcs, _ in rows:
        source, target = locate(importer.strip(' `')), locate(imported.strip(' `'))
        if source is None or target is None:
            bad.append(f'{importer.strip()} -> {imported.strip()}')
            continue
        bad += [f'{source.name}: {f}' for f in re.findall(r'`([^`]+)`', funcs) if not defined(source, f)]
    assert not bad, f'section 6 rows that do not match the code: {bad}'


def test_links_point_at_headings(texts):
    def slug(heading):
        return re.sub(r'[^a-z0-9_ -]', '', heading.lower()).replace(' ', '-')

    headings = {slug(h) for h in re.findall(r'^#+ (.+)$', texts['docs/ARCHITECTURE.md'], re.M)}
    anchors = re.findall(r'\]\(docs/ARCHITECTURE\.md#([^)]+)\)', texts['AGENTS.md'])
    assert anchors and not [a for a in anchors if a not in headings], f'anchors without a heading: {anchors}'


def test_documented_commands_are_in_the_command_table(texts):
    from sosfuse2.__main__ import COMMANDS

    main = paths.PKG / '__main__.py'
    assert main.is_file(), 'the front door __main__.py is missing'
    used = set()
    for text in (texts['AGENTS.md'], texts['docs/ARCHITECTURE.md'], start_here(texts['README.md'])):
        used |= set(re.findall(r'python -m sosfuse2 ([a-z_]+)', text))
    assert used and used <= COMMANDS.keys(), f'commands that __main__.py does not know: {sorted(used - COMMANDS.keys())}'


def test_every_environment_variable_in_the_code_is_documented(texts):
    names = set()
    for path in paths.PKG.rglob('*.py'):
        if 'tests' not in path.relative_to(paths.PKG).parts:  # names that only the tests use are not required
            names |= set(re.findall(r'SOSFUSE2_[A-Z0-9_]*', path.read_text(encoding='utf-8')))
    names = {n for n in names if not n.endswith('_')}  # SOSFUSE2_ and SOSFUSE2_KSPLIT_VERIFY_ are name fragments
    table = texts['docs/ENVIRONMENT.md']
    absent = sorted(n for n in names if not re.search(r'(?<![A-Z0-9_])%s(?![A-Z0-9_])' % n, table))
    assert names and not absent, f'docs/ENVIRONMENT.md lacks: {absent}'
