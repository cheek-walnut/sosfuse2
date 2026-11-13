"""Identity rules: which bytes of the package enter a signature or a cache key.

  input                       identity
  Python source (.py)         text without comments and docstrings (python_text)
  C/CUDA source (.cu .cpp .h) text after line splicing, without comments (c_text)
  state dict                  the dict without note and label keys at its top level (state_identity)
  state/config JSON file      the document without note keys, states as above, key order kept (json_text)
  registered numeric asset   content without its documented provenance fields (asset_identity)
  everything else             raw bytes

Python. Comments and docstrings (the first statement of a module, class or function body when it is a plain string
constant) are deleted; then trailing blanks are stripped and empty lines dropped. String literals are kept byte for
byte, including a comment written inside an f-string replacement field. An empty line stays where the line before it
ends with a continuation backslash, because dropping it would join two logical lines. Everything else, including
indentation and spacing inside a line, is identity.

C/CUDA. The translation phase order is followed: backslash-newline pairs are removed first, then each comment becomes
one space; then trailing blanks are stripped and empty lines dropped. String and character literals and the header
name of an #include are kept. A quote without its closing quote on the same line keeps the rest of that line.
Spacing inside a line is identity, so a comment added or removed between two tokens of a line changes the identity
unless one space stands in its place; a comment at the end of a line or on lines of its own never does.

JSON. Keys named description, note, notes, note_* or *_note are dropped, except below a key that holds inputs of the
computation (VERBATIM_KEYS: settings, parameters, stratosphere, density), whose value is kept whole. A value under the
key `state` (dict) or `states` (list of dicts) is a state: only its top-level note keys and STATE_LABELS are dropped
and nothing below is touched, so a state reads the same from a file and from memory. Key order is identity;
whitespace is not.

Not identity: line numbers, the amount of trailing whitespace, empty lines, the text of comments and docstrings.
Input outside the supported forms is refused with E_IDENTITY_SOURCE and is never hashed as raw bytes instead.
Normalized hashes are prefixed with RULES; a change of these rules must change RULES so that every key built on them
changes with it. Compiler build keys (the NVRTC cache, the native library) use raw bytes and do not come from here.
"""

import ast
import bisect
import codecs
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import tokenize
from pathlib import Path

from . import paths

RULES = 'sosfuse2-identity-2'

NOTE_KEYS = ('description', 'note', 'notes')
STATE_LABELS = ('group', 'role', 'holdout', 'out_of_axis_range', 'q2d_override')
VERBATIM_KEYS = ('settings', 'parameters', 'stratosphere', 'density')

SUFFIX_KINDS = {'.py': 'python', '.cu': 'c', '.cpp': 'c', '.h': 'c'}

_BOM = b'\xef\xbb\xbf'
_TRIGRAPH_TAILS = '=/\'()!<>-'
_RAW_STRING_PREFIXES = ('R', 'LR', 'uR', 'UR', 'u8R')
_HEADER_DIRECTIVES = ('include', 'include_next', 'import', 'embed')
_HEADER_OPERATORS = ('__has_include', '__has_include_next', '__has_embed')
_C_BLANK = ' \t\f\v'
_PY_BLANK = ' \t\f'


def _refused(name, reason):
    return ValueError(f'E_IDENTITY_SOURCE: {name}: {reason}')


def _utf8(raw, name):
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f'identity input must be bytes, got {type(raw).__name__}')
    raw = bytes(raw)
    if raw.startswith(_BOM):
        raise _refused(name, 'byte order mark')
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise _refused(name, f'not UTF-8 ({exc.reason} at byte {exc.start})') from None


def _source_text(raw, name):
    """UTF-8 text with '\\n' line ends. A carriage return that is not part of '\\r\\n' is refused."""
    text = _utf8(raw, name).replace('\r\n', '\n')
    if '\r' in text:
        raise _refused(name, 'carriage return without line feed')
    return text


def _docstring(node):
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first
    return None


def python_text(raw, name='<python>'):
    """Identity text of Python source bytes (rules in the module docstring). The result is not meant to be compiled:
    a body that was only a docstring is left empty."""
    text = _source_text(raw, name)
    try:
        encoding = tokenize.detect_encoding(io.BytesIO(bytes(raw)).readline)[0]
        codec = codecs.lookup(encoding).name
    except (SyntaxError, LookupError) as exc:
        raise _refused(name, f'encoding declaration: {exc}') from None
    if codec != 'utf-8':
        raise _refused(name, f'encoding declaration {encoding!r} is not utf-8')
    try:
        tree = ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError) as exc:
        raise _refused(name, f'{type(exc).__name__}: {exc}') from None

    lines = text.split('\n')
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    def at(position):
        row, column = position
        if not 1 <= row <= len(lines) or not 0 <= column <= len(lines[row - 1]):
            raise _refused(name, f'token position {position} outside the source')
        return starts[row - 1] + column

    def at_bytes(row, column):
        line = lines[row - 1]
        if not line.isascii():
            try:
                column = len(line.encode('utf-8')[:column].decode('utf-8'))
            except UnicodeDecodeError:
                raise _refused(name, f'node offset inside a character at line {row}') from None
        return at((row, column))

    # String literals: one span per literal, an f-string or t-string from its start token to its matching end token.
    strings = []
    comments = []
    open_starts = []
    for token in tokens:
        kind = tokenize.tok_name[token.type]
        if token.type == tokenize.COMMENT:
            a, b = at(token.start), at(token.end)
            if text[a:b] != token.string or not token.string.startswith('#') or text[b : b + 1] not in ('', '\n'):
                raise _refused(name, f'comment token at {token.start} does not match the source')
            comments.append((a, b))
        elif token.type == tokenize.STRING:
            if not open_starts:
                strings.append((at(token.start), at(token.end)))
        elif kind.endswith('STRING_START'):
            open_starts.append(at(token.start))
        elif kind.endswith('STRING_END'):
            if not open_starts:
                raise _refused(name, f'string end without start at {token.start}')
            a = open_starts.pop()
            if not open_starts:
                strings.append((a, at(token.end)))
    if open_starts:
        raise _refused(name, 'string start without end')
    for a, b in strings:
        if b - a < 2 or text[b - 1] not in '"\'' or not (text[a] in '"\'' or text[a].isalpha()):
            raise _refused(name, f'string token at offset {a} does not match the source')

    removed = []
    for node in ast.walk(tree):
        first = _docstring(node)
        if first is None:
            continue
        a, b = at_bytes(first.lineno, first.col_offset), at_bytes(first.end_lineno, first.end_col_offset)
        try:
            again = ast.parse(text[a:b], mode='eval').body
        except (SyntaxError, ValueError):
            again = None
        if not isinstance(again, ast.Constant) or again.value != first.value.value:
            raise _refused(name, f'docstring span at line {first.lineno} does not match the source')
        removed.append((a, b))
    removed.sort()
    doc_starts = [a for a, _ in removed]

    def in_docstring(a):
        i = bisect.bisect_right(doc_starts, a) - 1
        return i >= 0 and a < removed[i][1]

    kept_strings = [span for span in strings if not in_docstring(span[0])]
    string_starts = [a for a, _ in kept_strings]

    def in_string(a):
        i = bisect.bisect_right(string_starts, a) - 1
        return i >= 0 and a < kept_strings[i][1]

    # A comment inside a kept string span (an f-string replacement field) stays; one inside a docstring goes with it.
    removed += [span for span in comments if not in_string(span[0]) and not in_docstring(span[0])]
    spans = sorted([(a, b, False) for a, b in removed] + [(a, b, True) for a, b in kept_strings])

    out = []
    current = []

    def end_line():
        line = ''.join(current).rstrip(_PY_BLANK)
        current.clear()
        if line:
            out.append(line)
        elif out and out[-1].endswith('\\'):
            out.append('')  # the empty line ends the logical line that the backslash continued

    def code(piece):
        parts = piece.split('\n')
        current.append(parts[0])
        for part in parts[1:]:
            end_line()
            current.append(part)

    position = 0
    for a, b, keep in spans:
        if a < position:
            raise _refused(name, f'overlapping spans at offset {a}')
        code(text[position:a])
        if keep:
            current.append(text[a:b])  # a string literal: its own newlines do not end a line here
        position = b
    code(text[position:])
    end_line()
    return '\n'.join(out) + '\n'


def _c_literal_end(text, i):
    """Index after the literal that opens at text[i]; the end of the line when the closing quote is missing."""
    quote = text[i]
    n = len(text)
    j = i + 1
    while j < n:
        c = text[j]
        if c == '\n':
            return j
        if c == '\\' and j + 1 < n and text[j + 1] != '\n':
            j += 2
            continue
        j += 1
        if c == quote:
            return j
    return n


def _c_word(c):
    return c == '_' or c.isalnum() or ord(c) > 127


def c_text(raw, name='<c>'):
    """Identity text of C, C++ or CUDA source bytes (rules in the module docstring). Refused: raw string literals,
    trigraphs, a backslash separated from its newline by blanks, an unterminated block comment or header name, a
    directive spelled with the %: digraph, and __has_include."""
    text = _source_text(raw, name)
    k = text.find('??')
    while k >= 0:
        if text[k + 2 : k + 3] and text[k + 2] in _TRIGRAPH_TAILS:
            raise _refused(name, f'trigraph {text[k : k + 3]!r}')
        k = text.find('??', k + 1)
    k = text.find('\\')
    while k >= 0:
        j = k + 1
        while text[j : j + 1] and text[j] in _C_BLANK:
            j += 1
        if j > k + 1 and text[j : j + 1] in ('\n', ''):
            raise _refused(name, 'backslash separated from the line end by blanks')
        k = text.find('\\', k + 1)
    text = text.replace('\\\n', '')

    out = []
    n = len(text)
    i = 0
    START, HASH, HEADER, BODY = range(4)  # where the scan is inside the current line
    state = START
    while i < n:
        c = text[i]
        if c == '\n':
            out.append(c)
            i += 1
            state = START
        elif c in _C_BLANK:
            out.append(c)
            i += 1
        elif c == '/' and text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
            out.append(' ')
        elif c == '/' and text.startswith('/*', i):
            j = text.find('*/', i + 2)
            if j < 0:
                raise _refused(name, 'unterminated block comment')
            i = j + 2
            out.append(' ')
        elif c == '<' and state == HEADER:
            j = text.find('>', i)
            line_end = text.find('\n', i)
            if j < 0 or 0 <= line_end < j:
                raise _refused(name, 'unterminated header name')
            out.append(text[i : j + 1])
            i = j + 1
            state = BODY
        elif c in '"\'':
            j = _c_literal_end(text, i)
            out.append(text[i:j])
            i = j
            state = BODY
        elif c.isdigit() or (c == '.' and text[i + 1 : i + 2].isdigit()):
            j = i + 1
            while j < n:
                d = text[j]
                if d in 'eEpP' and text[j + 1 : j + 2] in ('+', '-'):
                    j += 2
                elif d == "'" and j + 1 < n and _c_word(text[j + 1]):
                    j += 2  # digit separator
                elif d == '.' or _c_word(d):
                    j += 1
                else:
                    break
            out.append(text[i:j])
            i = j
            state = BODY
        elif _c_word(c):
            j = i + 1
            while j < n and _c_word(text[j]):
                j += 1
            word = text[i:j]
            if word in _RAW_STRING_PREFIXES and text[j : j + 1] == '"':
                raise _refused(name, 'raw string literal')
            if word in _HEADER_OPERATORS:
                raise _refused(name, f'{word} is not supported')
            out.append(word)
            i = j
            state = HEADER if state == HASH and word in _HEADER_DIRECTIVES else BODY
        else:
            if state == START and text.startswith('%:', i):
                raise _refused(name, 'directive spelled with a digraph')
            out.append(c)
            i += 1
            state = HASH if state == START and c == '#' else BODY
    lines = (line.rstrip(_C_BLANK) for line in ''.join(out).split('\n'))
    return '\n'.join(line for line in lines if line) + '\n'


def is_note(key):
    """True for a key whose value is prose for readers and never an input of a computation."""
    return isinstance(key, str) and (key in NOTE_KEYS or key.startswith('note_') or key.endswith('_note'))


def state_identity(state):
    """Deep copy of a state dict without its top-level note keys and STATE_LABELS. Nested dicts are not touched and
    unknown keys stay, so a new key is identity until it is listed here."""
    return {key: copy.deepcopy(value) for key, value in state.items() if not is_note(key) and key not in STATE_LABELS}


def json_identity(doc):
    """Copy of a JSON document without its note keys. A dict under `state` and the dicts of a list under `states`
    become state_identity() of themselves and are not descended into; a value under one of VERBATIM_KEYS is copied
    whole. Key order is kept."""
    if isinstance(doc, dict):
        out = {}
        for key, value in doc.items():
            if is_note(key):
                continue
            if key in VERBATIM_KEYS:
                out[key] = copy.deepcopy(value)
            elif key == 'state' and isinstance(value, dict):
                out[key] = state_identity(value)
            elif key == 'states' and isinstance(value, list) and all(isinstance(item, dict) for item in value):
                out[key] = [state_identity(item) for item in value]
            else:
                out[key] = json_identity(value)
        return out
    if isinstance(doc, (list, tuple)):
        return [json_identity(item) for item in doc]
    return copy.deepcopy(doc)


def _json_document(raw, name):
    """Parse JSON without filtering, rejecting BOM, invalid UTF-8, duplicate keys and all non-finite numbers."""

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f'repeated key {key!r}')
            out[key] = value
        return out

    def constant(token):
        raise ValueError(f'non-finite number {token}')

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            constant(token)
        return value

    text = _utf8(raw, name)
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    except (ValueError, RecursionError) as exc:
        raise _refused(name, f'JSON: {exc}') from None


def json_text(raw, name='<json>'):
    """Strictly parse all values before removing documentary fields."""
    return json.dumps(json_identity(_json_document(raw, name)), separators=(',', ':'), allow_nan=False)


_TEXT = {'python': python_text, 'c': c_text, 'json': json_text}
_MEMO = {}  # (kind, sha256 of the raw bytes) -> identity sha256; never keyed by path or mtime


def raw_sha(path):
    """sha256 of the file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Only these asset fields are documentary. Unknown fields remain identity.
ASSET_NOTES = {
    'spectra_fine.json': ('definition', 'wavelength_reference', 'sources', 'generator', 'previous_sha256', 'self_check'),
    'spectra_fine_temperature.json': ('wavelength_reference', 'sources', 'generator', 'previous_sha256'),
    'temperature_usstd76.json': ('generator_sha256', 'formula_source'),
    'no2_afglus_profile.json': ('source', 'generator', 'semantics'),
    'no2_vandaele1998_fine.json': ('definition', 'wavelength_reference', 'self_check'),
    'o4_hitran2024_fine.json': ('definition', 'wavelength_reference', 'self_check'),
}
MINOR_ASSETS = ('no2_vandaele1998_fine.json', 'o4_hitran2024_fine.json')
# Raw digest -> content identity, measured from each published asset. Never infer equivalence from a filename.
ASSET_SUPERSEDED = {'spectra_fine.json': {'be7f9798ca8597b980273c12813834d159f720cd0211a3ca73d7f6715d5ac405': '452376079c9d9112b26ce653220045fd87efe37cef1cc5c61cac3261dc0e7622'},
 'spectra_fine_temperature.json': {'a809a5423c3752b5c19f4dd30b55bc1b4bb11a9ce146180579e0f1344dbd618b': '335b9ef0d38e880412c07a53bd1a7c6cf8afb0b40403799e1f9ce076a971c066'},
 'temperature_usstd76.json': {'98e83695b49ee3d183d4623281b0bad7669227f561de24f330d14afd655a49b8': 'c0e352ab1629f7d60db588dff7704d960083cce628e01f599252329096042927'},
 'no2_afglus_profile.json': {'52a09edf32a06f40149af67dde884313b4e780a38b0899a1c2fcee98005df029': 'f1a254c5f5710eff273c1d3989c277d656e7c03d07cdac020fa7fb2e8d29b928'},
 'no2_vandaele1998_fine.json': {'23bdb5fb1f8a2e99f968ad2d3ce8395c9e3190c0aa8fb2e30289beea37f0fb9e': '1a145d78ed28ac3f8dee88606ef29273355dc1ae06018255ee21f1fb0cdc4c5d'},
 'o4_hitran2024_fine.json': {'dea296e1303fb10704354224c1d9c879730e7b8f2ebda3ce8bafe0481c9454a5': '608097ca6536387dbe4b16dc8cf06c270c7f7e07cb83487298e01f8e63f7ea60'}}


def asset_digest(name, digest):
    """Canonicalize a registered raw digest; unknown digests are preserved for exact comparison."""
    return ASSET_SUPERSEDED.get(name, {}).get(digest, digest) if isinstance(digest, str) else digest


def asset_identity(raw, name):
    """Content hash of numeric JSON bytes; raw_sha remains the byte-integrity hash.

    Arrays, units, formats and fields outside the documentary subtrees remain signed. Source descriptions and generator byte hashes do not
    describe the resulting computation. Minor-absorber source pins and generator rules_version remain signed.
    The frozen spectral reference is canonicalized, not discarded.
    """
    doc = _json_document(raw, name)
    for key in ASSET_NOTES[name]:
        doc.pop(key, None)
    if name in MINOR_ASSETS:
        gen = doc.get('generator', {})
        doc['generator'] = {k: v for k, v in gen.items() if k not in ('sha256', 'identity_sha256')}
        source = doc.get('source', {})
        doc['source'] = {k: v for k, v in source.items() if k not in (
            'name', 'dataset', 'page', 'licence', 'reference', 'references', 'url', 'wavelength', 'unit_conversion',
        )}
    if 'frozen_source_sha256' in doc:
        doc['frozen_source_sha256'] = asset_digest('spectra_fine.json', doc['frozen_source_sha256'])
    text = json.dumps(doc, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(f'{RULES}:asset-1:{name}\n{text}'.encode()).hexdigest()


def asset_sha(path):
    """Content identity for a registered numeric asset; raw SHA for other data files."""
    path = Path(path)
    raw = path.read_bytes()
    return asset_identity(raw, path.name) if path.name in ASSET_NOTES else hashlib.sha256(raw).hexdigest()


def asset_matches(name, recorded, raw):
    """Accept the actual bytes, their content identity, or a registered equivalent publication."""
    return recorded == hashlib.sha256(raw).hexdigest() or asset_digest(name, recorded) == asset_identity(raw, name)


def file_sha(path, kind=None):
    """Identity sha256 of a file. `kind` is 'python', 'c', 'json' or 'raw'; None picks it from the suffix
    (SUFFIX_KINDS, anything else is 'raw'). 'json' is never picked from the suffix: callers name states/settings;
    registered numeric assets use asset_sha instead.
    'raw' is the plain sha256 of the bytes; the other kinds hash RULES, the kind and the normalized text."""
    path = Path(path)
    if kind is None:
        kind = SUFFIX_KINDS.get(path.suffix, 'raw')
    if kind != 'raw' and kind not in _TEXT:
        raise ValueError(f'E_IDENTITY_KIND: unknown kind {kind!r}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if kind == 'raw':
        return digest
    key = (kind, digest)
    if key not in _MEMO:
        text = _TEXT[kind](raw, paths.rel(path))
        _MEMO[key] = hashlib.sha256(f'{RULES}:{kind}\n'.encode() + text.encode('utf-8')).hexdigest()
    return _MEMO[key]


# Fixed provenance inputs beside the production modules: the engine module, the host helpers and the data the engine
# reads. Adding a file here changes every signature (intended); a listed file that is missing is an error.
ACCEPTED_SOURCES = ('__init__.py', 'engine.py')
HOST_MODULES = ('__init__.py', 'host_cores.py', 'pack_state.py')
DATA_ASSETS = (
    'spectra.json',
    'spectra_fine.json',
    'spectra_fine_temperature.json',
    'lef_cie.json',
    'cmf_1nm.json',
    'states_v1.json',
    'states_v2.json',
    'states_v1_usstd.json',
    'rgb_weights.npy',
    'temperature_usstd76.json',
    'no2_afglus_profile.json',
    'no2_vandaele1998_fine.json',
)
STATE_ASSETS = ('states_v1.json', 'states_v2.json', 'states_v1_usstd.json')  # hashed as 'json'


def package_hashes():
    """Code and data hashes of the provenance format (docs/PROVENANCE_V3.md): package-relative keys ('sosfuse2/...')
    over the production modules (paths.PRODUCTION_MODULES, a fixed list), the CUDA kernel, the engine module, the
    host helpers, the native CPU kernel source and the data assets the engine reads. Values are file_sha(): sources
    and state files by their identity text, registered numeric assets by asset_sha. Every entry is a fixed name and a
    missing file is E_PACKAGE_FILE_MISSING, not a gap."""
    P = paths.PKG
    sources = paths.production_files()
    sources += [
        P / 'kernel.cu',
        *[P / 'accepted' / n for n in ACCEPTED_SOURCES],
        *[P / 'host' / n for n in HOST_MODULES],
        P / 'native/sh_cpu.cpp',
    ]
    assets = [P / 'assets' / n for n in DATA_ASSETS]
    missing = [paths.rel(p) for p in sources + assets if not p.is_file()]
    if missing:
        raise RuntimeError(f'E_PACKAGE_FILE_MISSING: provenance inputs absent from the package: {missing}')
    hashes = {paths.rel(p): file_sha(p) for p in sources}
    hashes.update({paths.rel(p): asset_sha(p) if p.name in ASSET_NOTES
                   else file_sha(p, 'json' if p.name in STATE_ASSETS else 'raw') for p in assets})
    return hashes


def package_info():
    """Release identity beside the content hashes: version from the installed metadata (None in-tree without install); commit from
    SOSFUSE2_PACKAGE_COMMIT, else the installer's direct_url.json (pip/uv VCS install: vcs_info.commit_id; used only when the
    distribution's sosfuse2/__init__.py is this package's own), else the git checkout whose
    src/sosfuse2 IS this package (never a consumer checkout that merely holds an installed copy). None otherwise, never an error."""
    dist = None
    ver = None
    try:
        from importlib.metadata import distribution

        dist = distribution('sosfuse2')
        ver = dist.version
    except Exception:
        pass
    commit = os.environ.get('SOSFUSE2_PACKAGE_COMMIT') or None
    if commit is None and dist is not None:
        try:
            # Only when the metadata describes THIS copy of the package: an in-tree run (PYTHONPATH=src) next to an installed
            # sosfuse2 must not record the installed copy's commit.
            if Path(dist.locate_file('sosfuse2/__init__.py')).resolve() == (Path(paths.PKG) / '__init__.py').resolve():
                du = dist.read_text('direct_url.json')
                commit = ((json.loads(du).get('vcs_info') or {}).get('commit_id') or None) if du else None
        except Exception:
            commit = None
    if commit is None:
        try:
            top = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'], cwd=paths.PKG, capture_output=True, text=True, timeout=10
            )
            if (
                top.returncode == 0
                and (Path(top.stdout.strip()) / 'src/sosfuse2').resolve() == Path(paths.PKG).resolve()
            ):
                r = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], cwd=paths.PKG, capture_output=True, text=True, timeout=10
                )
                commit = r.stdout.strip() or None if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return dict(package_version=ver, package_commit=commit)
