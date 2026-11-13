"""Shared bake cache store: one copy per content key.

Store  : <rclone remote>:<path>/{geometry,attach,columns}/<the key names the pipeline already uses>; the remote is named by
         SKY_CACHE_REMOTE or the remote= argument, never built in (the remote work folder is the original, local copies are caches).
Mirror : SKY_CACHE_MIRROR (default <work root>/sky-cache)/{geometry,attach,columns}; callers may pass it as --geometry-cache,
         --column-cache and STRATO_ATTACH_CACHE (cold runs such as gate.py --gf keep their own empty caches).

  python -B -m sosfuse2.cache_store list [--json]
  python -B -m sosfuse2.cache_store push <local_root>                 complete units only, never overwrites
  python -B -m sosfuse2.cache_store pull (--all | --keys K ...) [--legacy] <local_root>
  python -B -m sosfuse2.cache_store pack-for-rig [--local DIR] (--all | K ...) <out.tar>

A unit is what is checked complete, transferred and never overwritten. Local layout = the pipeline's (paths relative
to the root); in the store every unit is ONE object, so an interrupted upload leaves nothing that looks complete:
  geometry/endpoints_<20 hex>   local: identity.json, source_height.json, packed/* (complete = packed/complete.json,
      layout.json and every packed/<field>.bin of the size layout.json names, geometry.py; users/, *.lock,
      *.tmp never belong to it). Store: geometry/endpoints_<sig>.tar, MANIFEST.json first ([rel, bytes, sha256] per
      file + content_sha256), deterministic members. The signature leaves the scale height out
      (geometry_contract.py) while end_col[:,1] and source_height.json come from the height that built it, so two
      valid local copies of one key can differ: a store tar whose manifest differs from the local copy is
      E_CACHE_CONFLICT (kept, never overwritten or combined). Pull streams the tar, checks every member against the
      manifest and publishes the folder under the builder lock.
  geometry/endpoints_<sig>/aerosol/<name>.f4  per-height column (tmp + os.replace, geometry_cpu.py); needs its endpoints.
  geometry/endpoints_<sig>/host_cache/<64 hex>.hcache  one-file envelope (host_cache.py): header size check.
  geometry/host_sun/.../sun_columns.npz      sun table (tmp + os.replace, sun_parallel.py): zip directory check.
  attach/attach_geometry_<24 hex>.npz        published by atomic rename only (run_gpu_state_v3.py,
      stratosphere.py): zip directory check. attach/attach_<16 hex>.done (group markers, a dedup hint:
      run_gpu_state_v3.py) travel after every npz.
  columns/<64 hex>   <key>.npy + marker <key>.sha256 = sha256(npy) (column_cache.py). The marker is attached only
      to an npy this push uploaded or a stored npy whose bytes were downloaded and hashed equal; a stored marker that
      differs is E_CACHE_CONFLICT.
  Single-file units already stored with another size are E_CACHE_CONFLICT (same size: kept, contents not compared).
Legacy: earlier stores used loose endpoint files
(geometry/endpoints_<sig>/{identity.json,source_height.json,packed/*}). That form is never written again and is read
only by `pull --legacy` when no tar of the signature exists; `list` marks it. Nothing is deleted.
Pull downloads into <root>/.cache_store/pull-* (same file system), re-checks every unit there, then publishes with the
pipeline's own locks or no-clobber hard links, marker last; it never replaces a file the mirror already has.
Concurrency: endpoints folders are leased under .v3-leases.lock and built under <sig>.lock
(run_gpu_state_v3.py), and complete.json is written last (geometry.py). A caller sharing geometry must retain it;
aerosol columns, host sidecars and sun tables take per-file locks and publish by
os.replace (geometry_cpu.py, host_cache.py); attach files are computed in a private overlay and
renamed in (run_gpu_state_v3.py); columns are read and written under a per-key lock (column_cache.py),
first-use markers stay in each job's work root; the geometry-only prefetch attaches through the same overlay.
flock needs a local file system (keep the mirror on a local disk).
Failures: CacheStoreError or row codes with stable E_CACHE_* names; rclone output is cut to its tail with URLs and
token-like values removed."""

import argparse, contextlib, errno, fcntl, hashlib, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time, uuid, zipfile
from pathlib import Path
from sosfuse2 import paths

# No built-in store: the remote, the rclone config and the mirror are the operator's.
# REMOTE None -> E_CACHE_REMOTE_MISSING when a store operation needs it; CONFIG None -> rclone's own default config file;
# MIRROR None -> <work root>/sky-cache (mirror()).
REMOTE = os.environ.get('SKY_CACHE_REMOTE') or None
CONFIG = os.environ.get('SKY_CACHE_RCLONE_CONFIG') or None
MIRROR = os.environ.get('SKY_CACHE_MIRROR') or None


def mirror():
    return Path(MIRROR) if MIRROR else paths.work_root() / 'sky-cache'


def _remote(remote):
    r = remote or REMOTE
    if not r:
        raise CacheStoreError('E_CACHE_REMOTE_MISSING', 'no cache store remote: pass remote= or set SKY_CACHE_REMOTE')
    return r


KINDS = ('geometry', 'attach', 'columns')
STAGE = '.cache_store'  # staging folder inside a local root (same file system)
ITEMSIZE = dict(i8=8, f8=8, u8=8, f4=4, i4=4, u4=4, u2=2, i2=2, u1=1, i1=1)
FORMAT = 'sky-cache-endpoints/1'
MANIFEST = 'MANIFEST.json'
HEAD = 1 << 20  # bytes read to get a stored tar's manifest (first member)
DISK_MARGIN = 64 << 20
# Publication order: an endpoints unit before the units that need it, attach markers after every attach npz.
ORDER = ('endpoints', 'aerosol', 'hcache', 'sun', 'attach', 'columns', 'attach_done')

ENDPOINTS = re.compile(r'geometry/endpoints_[0-9a-f]{20}')
ENDPOINTS_TAR = re.compile(r'(geometry/endpoints_[0-9a-f]{20})\.tar')
LEGACY = re.compile(r'(geometry/endpoints_[0-9a-f]{20})/(?:identity\.json|source_height\.json|packed/[^/]+)')
AEROSOL = re.compile(r'(geometry/endpoints_[0-9a-f]{20})/aerosol/[^/]+\.f4')
HCACHE = re.compile(r'geometry/endpoints_[0-9a-f]{20}/host_cache/[0-9a-f]{64}\.hcache')
SUN = re.compile(r'geometry/host_sun/(?:[^/]+/)+sun_columns\.npz')
ATTACH = re.compile(r'attach/attach_geometry_[0-9a-f]{24}\.npz')
ATTACH_DONE = re.compile(r'attach/attach_[0-9a-f]{16}\.done')
COLUMN = re.compile(r'columns/([0-9a-f]{64})\.(npy|sha256)')


class CacheStoreError(RuntimeError):
    def __init__(self, code, detail=''):
        super().__init__(f'{code}: {detail}' if detail else code)
        self.code = code
        self.detail = detail


class Unit:
    """files/markers: paths relative to a root; fmt 'files' (local or single objects), 'tar' (stored endpoints),
    'legacy' (endpoints stored as loose files by the first fill, read only)."""

    __slots__ = ('key', 'kind', 'files', 'markers', 'requires', 'bytes', 'fmt')

    def __init__(self, key, kind, files, markers=(), requires=None, size=0, fmt='files'):
        self.key, self.kind, self.files, self.markers, self.requires, self.bytes, self.fmt = (
            key,
            kind,
            tuple(files),
            tuple(markers),
            requires,
            size,
            fmt,
        )

    def all(self):
        return self.files + self.markers

    def row(self):
        return dict(
            key=self.key,
            kind=self.kind + ('(legacy)' if self.fmt == 'legacy' else ''),
            files=len(self.all()),
            bytes=self.bytes,
        )


def say(**kw):
    print('CACHE_STORE ' + json.dumps(kw, default=str), flush=True)


def redact_connections(text, *, remote=False):
    """Hide all rclone connection parameters, retaining the backend and remote path."""
    text = str(text)
    start = re.compile(r'(?<![\w.+@-])(:?[\w.+@-]+(?: [\w.+@-]+)*),(?=[\w.-]+(?:=|,|:))')
    parts = []
    pos = 0
    while match := (start.match(text, pos) if remote else start.search(text, pos)):
        outer = text[match.start() - 1] if match.start() and text[match.start() - 1] in "'\"" else None
        parts.extend((text[pos : match.end()], '<redacted>'))
        i = match.end()
        quote = None
        while i < len(text):
            c = text[i]
            if quote:
                if c == quote:
                    if i + 1 < len(text) and text[i + 1] == quote:
                        i += 1
                    else:
                        quote = None
            elif c in "'\"":
                quote = c
            elif c == ':' and text[i + 1 : i + 3] != '//':
                break
            i += 1
        if remote:
            parts.append(text[i:])
            return ''.join(parts)
        end = i + 1 if i < len(text) else i
        if outer:
            while end < len(text) and text[end] != outer:
                end += 1
        else:
            while end < len(text) and not text[end].isspace() and text[end] not in "'\"":
                end += 1
        parts.append(text[i:end])
        pos = end
    parts.append(text[pos:])
    return ''.join(parts)


def clean(text, limit=600):
    """rclone output for a record: tail only, URLs and token-like values removed (signed URLs and OAuth tokens are
    secrets until they expire)."""
    text = redact_connections(text)
    text = re.sub(r'(?i)\b[a-z][a-z0-9+.-]*://\S+', '<url>', text)
    text = re.sub(r'(?i)\b(bearer|basic)\s+\S+', r'\1 <redacted>', text)
    text = re.sub(r'\bya29\.[\w.-]+', '<redacted>', text)  # Google OAuth access token
    text = re.sub(
        r'(?i)(?<![\w.-])([a-z0-9_.-]*secret[a-z0-9_.-]*|[a-z0-9_.-]*_key(?:_id)?|'
        r'access_token|refresh_token|token|bearer|authorization|signature|x-amz-[a-z-]+|key|password)'
        r'''(["']?\s*[:=]\s*|\s+)(?:"(?:\\.|""|[^"\\\n])*(?:"|(?=\n|$))|'''
        r"""'(?:\\.|''|[^'\\\n])*(?:'|(?=\n|$))|[^\s"',}\]]+)""",
        r'\1=<redacted>',
        text,
    )
    return text[-limit:]


def _sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _zip_ok(path, members):
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except (OSError, zipfile.BadZipFile, ValueError):
        return False
    return set(members) <= names


def _hcache_ok(path):
    try:
        with open(path, 'rb') as f:
            line = f.readline(1 << 20)
        header = json.loads(line)
        return os.path.getsize(path) == len(line) + int(header['size'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _endpoints(root, key):
    """(Unit, None) if the endpoints folder is a complete build, else (None, reason)."""
    d = root / key
    packed = d / 'packed'
    for name in ('identity.json', 'source_height.json', 'packed/complete.json', 'packed/layout.json'):
        if not (d / name).is_file():
            return None, 'incomplete: no ' + name
    try:
        layout = json.loads((packed / 'layout.json').read_text())
        for field, shape in layout['shapes'].items():
            n = 1
            for x in shape:
                n *= int(x)
            if (packed / (field + '.bin')).stat().st_size != n * ITEMSIZE[layout['dtypes'][field]]:
                return None, 'incomplete: size ' + field
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f'incomplete: layout {type(exc).__name__}'
    files = ['identity.json', 'source_height.json'] + sorted(
        'packed/' + p.name
        for p in packed.iterdir()
        if p.is_file()
        and not p.is_symlink()
        and p.name != 'complete.json'
        and not p.name.endswith(('.lock', '.tmp'))
        and '.tmp' not in p.name
    )
    rel = [f'{key}/{f}' for f in files]
    size = sum((root / f).stat().st_size for f in rel) + (packed / 'complete.json').stat().st_size
    return Unit(key, 'endpoints', rel, [f'{key}/packed/complete.json'], size=size), None


def scan(root):
    """Complete units under a local root: ({key: Unit}, [(path, reason)] for what was left out)."""
    root = Path(root)
    units = {}
    skipped = []
    if not root.is_dir():
        raise CacheStoreError('E_CACHE_LOCAL_ROOT', f'not a directory: {root}')
    for kind in KINDS:
        top = root / kind
        if not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            here = Path(dirpath)
            rel_dir = here.relative_to(root).as_posix()
            if rel_dir == 'geometry':
                for name in [n for n in dirnames if n.startswith('endpoints_')]:
                    key = f'geometry/{name}'
                    if not ENDPOINTS.fullmatch(key):
                        skipped.append((key, 'name'))
                        continue
                    unit, why = _endpoints(root, key)
                    if unit:
                        units[key] = unit
                    else:
                        skipped.append((key, why))
            # never descend into lease markers, attach overlays or our own staging
            dirnames[:] = sorted(
                n for n in dirnames if n not in ('users', 'packed', STAGE) and not n.startswith('overlay-')
            )
            for name in sorted(filenames):
                rel = f'{rel_dir}/{name}'
                path = here / name
                if name.endswith(('.lock', '.tmp', '.partial')) or '.tmp.' in name:
                    skipped.append((rel, 'lock/tmp'))
                    continue
                if path.is_symlink():
                    skipped.append((rel, 'symlink'))
                    continue
                if rel_dir.startswith('geometry/endpoints_') and rel_dir.count('/') == 1:
                    continue  # identity/source_height: part of the endpoints unit
                if m := AEROSOL.fullmatch(rel):
                    units[rel] = Unit(rel, 'aerosol', [rel], requires=m.group(1), size=path.stat().st_size)
                elif HCACHE.fullmatch(rel):
                    if _hcache_ok(path):
                        units[rel] = Unit(rel, 'hcache', [rel], size=path.stat().st_size)
                    else:
                        skipped.append((rel, 'incomplete: envelope'))
                elif SUN.fullmatch(rel):
                    if _zip_ok(path, ('h.npy', 'x.npy', 'col.npy')):
                        units[rel] = Unit(rel, 'sun', [rel], size=path.stat().st_size)
                    else:
                        skipped.append((rel, 'incomplete: zip'))
                elif ATTACH.fullmatch(rel):
                    if _zip_ok(path, ('offsets.npy', 'suncol.npy')):
                        units[rel] = Unit(rel, 'attach', [rel], size=path.stat().st_size)
                    else:
                        skipped.append((rel, 'incomplete: zip'))
                elif ATTACH_DONE.fullmatch(rel):
                    units[rel] = Unit(rel, 'attach_done', [], [rel], size=path.stat().st_size)
                elif m := COLUMN.fullmatch(rel):
                    if m.group(2) == 'sha256':
                        continue
                    key = f'columns/{m.group(1)}'
                    side = here / (m.group(1) + '.sha256')
                    try:
                        good = side.is_file() and side.read_text().strip() == _sha_file(path)
                    except OSError:
                        good = False
                    if good:
                        units[key] = Unit(
                            key, 'columns', [rel], [key + '.sha256'], size=path.stat().st_size + side.stat().st_size
                        )
                    else:
                        skipped.append((rel, 'incomplete: checksum'))
                else:
                    skipped.append((rel, 'unknown'))
    return units, skipped


def group_remote(files, dups=()):
    """Units of a store listing {rel path: bytes}. Endpoints: the tar object (fmt 'tar'), else a complete legacy loose
    folder (fmt 'legacy'). Units touching a duplicated object (Drive allows two files of one name) are left out."""
    units = {}
    legacy = {}
    for rel in sorted(files):
        if m := ENDPOINTS_TAR.fullmatch(rel):
            units[m.group(1)] = Unit(m.group(1), 'endpoints', [rel], size=files[rel], fmt='tar')
        elif m := LEGACY.fullmatch(rel):
            legacy.setdefault(m.group(1), []).append(rel)
        elif m := AEROSOL.fullmatch(rel):
            units[rel] = Unit(rel, 'aerosol', [rel], requires=m.group(1), size=files[rel])
        elif HCACHE.fullmatch(rel):
            units[rel] = Unit(rel, 'hcache', [rel], size=files[rel])
        elif SUN.fullmatch(rel):
            units[rel] = Unit(rel, 'sun', [rel], size=files[rel])
        elif ATTACH.fullmatch(rel):
            units[rel] = Unit(rel, 'attach', [rel], size=files[rel])
        elif ATTACH_DONE.fullmatch(rel):
            units[rel] = Unit(rel, 'attach_done', [], [rel], size=files[rel])
        elif (m := COLUMN.fullmatch(rel)) and m.group(2) == 'npy':
            key = f'columns/{m.group(1)}'
            side = key + '.sha256'
            if side in files:
                units[key] = Unit(key, 'columns', [rel], [side], size=files[rel] + files[side])
    for key, rels in legacy.items():
        marker = key + '/packed/complete.json'
        if key in units or not {
            marker,
            key + '/identity.json',
            key + '/source_height.json',
            key + '/packed/layout.json',
        } <= set(rels):
            continue
        units[key] = Unit(
            key, 'endpoints', [f for f in rels if f != marker], [marker], size=sum(files[f] for f in rels), fmt='legacy'
        )
    return {k: u for k, u in units.items() if not set(u.all()) & set(dups)}


def select(units, keys):
    """keys None = all; else each key names a unit or a folder prefix ('geometry/endpoints_<sig>' takes its aerosol
    and host_cache units too). A key that matches nothing is E_CACHE_KEY_UNKNOWN."""
    if keys is None:
        return dict(units)
    out = {}
    for k in keys:
        k = k.strip('/').removesuffix('.tar')
        hit = {
            u: units[u]
            for u in units
            if u == k or u.startswith(k + '/') or (k.startswith('columns/') and u == k.removesuffix('.npy'))
        }
        if not hit:
            raise CacheStoreError('E_CACHE_KEY_UNKNOWN', k)
        out.update(hit)
    return out


def ordered(units):
    return sorted(units.values(), key=lambda u: (ORDER.index(u.kind), u.key))


# ---------------------------------------------------------------- endpoints tar (one object per key)


def manifest(root, u):
    """Content manifest of a complete local endpoints unit: [rel inside the key, bytes, sha256] per file."""
    rows = [[f[len(u.key) + 1 :], (Path(root) / f).stat().st_size, _sha_file(Path(root) / f)] for f in u.all()]
    rows.sort(key=lambda r: (r[0] == 'packed/complete.json', r[0]))
    content = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return dict(format=FORMAT, key=u.key, files=rows, content_sha256=content)


def _info(name, size):
    t = tarfile.TarInfo(name)
    t.size = size
    t.mode = 0o644
    t.mtime = 0
    t.uid = t.gid = 0
    t.uname = t.gname = ''
    return t


def write_tar(root, u, m, out):
    """Deterministic tar: MANIFEST.json, then the files in manifest order; each file re-hashed while written
    (E_CACHE_LOCAL_CHANGED if it no longer matches)."""
    body = (json.dumps(m, indent=1) + '\n').encode()
    with tarfile.open(out, 'w', format=tarfile.USTAR_FORMAT) as t:
        t.addfile(_info(MANIFEST, len(body)), io.BytesIO(body))
        for rel, size, digest in m['files']:
            h = hashlib.sha256()

            class Hashing(io.RawIOBase):
                def __init__(self, f):
                    self.f = f

                def read(self, n=-1):
                    b = self.f.read(n)
                    h.update(b)
                    return b

            with open(Path(root) / u.key / rel, 'rb') as f:
                t.addfile(_info(rel, size), Hashing(f))
            if h.hexdigest() != digest:
                raise CacheStoreError('E_CACHE_LOCAL_CHANGED', f'{u.key}/{rel} changed while packed')


MEMBER = re.compile(r'(identity\.json|source_height\.json|packed/[A-Za-z0-9_.-]+)')


def _manifest_json(data):
    """MANIFEST.json bytes -> manifest; format, member names and content_sha256 checked."""
    try:
        m = json.loads(data)
    except ValueError:
        raise CacheStoreError('E_CACHE_MANIFEST', 'manifest is not JSON') from None
    try:
        names = [r[0] for r in m['files']]
    except (KeyError, TypeError, IndexError):
        raise CacheStoreError('E_CACHE_MANIFEST', 'manifest files') from None
    if (
        m.get('format') != FORMAT
        or hashlib.sha256(json.dumps(m.get('files')).encode()).hexdigest() != m.get('content_sha256')
        or not all(isinstance(n, str) and MEMBER.fullmatch(n) and '..' not in n for n in names)
        or len(set(names)) != len(names)
    ):
        raise CacheStoreError('E_CACHE_MANIFEST', 'manifest format, names or content_sha256 mismatch')
    return m


def read_manifest(head):
    """Manifest from the first bytes of a stored endpoints tar (its first member)."""
    try:
        with tarfile.open(fileobj=io.BytesIO(head), mode='r|') as t:
            first = t.next()
            if first is None or first.name != MANIFEST or not first.isfile():
                raise ValueError('first member')
            data = t.extractfile(first).read()
    except (tarfile.TarError, ValueError, OSError) as exc:
        raise CacheStoreError('E_CACHE_MANIFEST', f'unreadable manifest ({type(exc).__name__})') from None
    return _manifest_json(data)


def remote_manifest(remote, key):
    return read_manifest(rclone(['cat', '--head', str(HEAD), f'{remote}/{key}.tar'], binary=True).stdout)


def extract_tar(stream, key, dst):
    """Stored endpoints tar -> dst/<key>/..., every member checked against the manifest (name, size, sha256) before
    the caller publishes it. Raises CacheStoreError E_CACHE_MANIFEST / E_CACHE_STAGED_INCOMPLETE."""
    seen = set()
    want = None
    try:
        with tarfile.open(fileobj=stream, mode='r|') as t:
            for i, member in enumerate(t):
                if i == 0:
                    if member.name != MANIFEST or not member.isfile():
                        raise CacheStoreError('E_CACHE_MANIFEST', 'first member')
                    m = _manifest_json(t.extractfile(member).read())
                    if m.get('key') != key:
                        raise CacheStoreError('E_CACHE_MANIFEST', 'manifest of another key')
                    want = {rel: (size, digest) for rel, size, digest in m['files']}
                    continue
                if want is None:
                    raise CacheStoreError('E_CACHE_MANIFEST', 'no manifest')
                rel = member.name
                if rel not in want or rel in seen or not member.isfile() or member.size != want[rel][0]:
                    raise CacheStoreError('E_CACHE_STAGED_INCOMPLETE', f'unexpected member {rel[:80]}')
                out = dst / key / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                src = t.extractfile(member)
                with open(out, 'wb') as f:
                    for block in iter(lambda: src.read(1 << 22), b''):
                        h.update(block)
                        f.write(block)
                if h.hexdigest() != want[rel][1]:
                    raise CacheStoreError('E_CACHE_STAGED_INCOMPLETE', f'sha256 {rel[:80]}')
                seen.add(rel)
            if want is None or seen != set(want):
                raise CacheStoreError('E_CACHE_STAGED_INCOMPLETE', f'{len(seen)} of {len(want or ())} members')
    except tarfile.TarError as exc:
        raise CacheStoreError('E_CACHE_STAGED_INCOMPLETE', f'tar {type(exc).__name__}') from None
    return m


# ---------------------------------------------------------------- rclone


def rclone_exe():
    exe = os.environ.get('SKY_CACHE_RCLONE') or shutil.which('rclone') or str(Path('~/.local/bin/rclone').expanduser())
    if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
        raise CacheStoreError('E_CACHE_RCLONE_MISSING', 'rclone not found')
    return exe


def rclone_cmd(args):
    if CONFIG is None:
        return [rclone_exe(), *args]  # rclone's default config file
    if not Path(CONFIG).is_file():
        raise CacheStoreError('E_CACHE_CONFIG_MISSING', 'rclone config not found')
    return [rclone_exe(), '--config', CONFIG, *args]


def rclone(args, ok=(0,), timeout=7200, binary=False):
    cmd = rclone_cmd(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=not binary, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CacheStoreError('E_CACHE_RCLONE_TIMEOUT', f'{args[0]} after {timeout} s') from None
    except OSError as exc:
        raise CacheStoreError('E_CACHE_RCLONE_MISSING', type(exc).__name__) from None
    if r.returncode not in ok:
        err = r.stderr.decode(errors='replace') if binary else (r.stderr or r.stdout)
        raise CacheStoreError('E_CACHE_RCLONE', f'{args[0]} rc={r.returncode} {clean(err)}')
    return r


def remote_listing(remote=None):
    """({rel path: bytes}, {rel paths listed more than once}); a store folder that does not exist yet is empty."""
    remote = _remote(remote)
    r = rclone(['lsjson', '-R', '--files-only', remote], ok=(0, 3))
    if r.returncode == 3:
        if 'directory not found' in (r.stderr + r.stdout):
            return {}, set()
        raise CacheStoreError('E_CACHE_RCLONE', f'lsjson rc=3 {clean(r.stderr or r.stdout)}')
    try:
        rows = json.loads(r.stdout or '[]')
    except ValueError:
        raise CacheStoreError('E_CACHE_LIST_FORMAT', 'lsjson output is not JSON') from None
    files, dups = {}, set()
    for row in rows:
        if row['Path'].split('/', 1)[0] not in KINDS:
            continue
        if row['Path'] in files:
            dups.add(row['Path'])
        files[row['Path']] = int(row['Size'])
    return files, dups


def remote_files(remote=None):
    return remote_listing(remote)[0]


def copy_files(src, dst, rels, work):
    """rclone copy of exactly these relative paths (unit files only: never a lock, tmp or lease marker); never
    replaces a destination file (--ignore-existing). rclone refuses other filter flags next to --files-from-raw."""
    if not rels:
        return 0.0
    bad = [r for r in rels if r.endswith(('.lock', '.tmp')) or '/users/' in r]
    if bad:
        raise CacheStoreError('E_CACHE_FILTER', f'{len(bad)} lock/tmp/lease paths in a transfer list')
    work.mkdir(parents=True, exist_ok=True)
    listing = work / f'files-{uuid.uuid4().hex[:8]}.txt'
    listing.write_text(''.join(r + '\n' for r in rels))
    t0 = time.monotonic()
    try:
        rclone(
            ['copy', '--files-from-raw', str(listing), '--ignore-existing', '--no-traverse', '-q', str(src), str(dst)]
        )
    finally:
        listing.unlink(missing_ok=True)
    return time.monotonic() - t0


def stream_tar(remote, key, dst):
    """`rclone cat <key>.tar` piped into extract_tar (no temporary tar). Returns the manifest."""
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(rclone_cmd(['cat', f'{remote}/{key}.tar']), stdout=subprocess.PIPE, stderr=err)
        failure = None
        try:
            m = extract_tar(p.stdout, key, dst)
            p.stdout.read()  # drain (tar end blocks)
        except Exception as e:
            failure = e
        finally:
            p.stdout.close()
            rc = p.wait()
        # A failed download is reported as such, even when it made the tar unreadable (rclone reason kept).
        if rc:
            err.seek(0)
            raise CacheStoreError(
                'E_CACHE_RCLONE', f'cat rc={rc} {clean(err.read().decode(errors="replace"))}'
            ) from failure
        if failure is not None:
            raise failure
    return m


# ---------------------------------------------------------------- commands


def _work(work):
    return Path(work) if work else paths.work_root() / 'tmp'


def push(root, remote=None, work=None):
    """Upload the complete units of root that the store lacks: endpoints as one tar each, then single files, then
    markers. Never overwrites; a stored unit that differs from the local one is reported, not combined."""
    root = Path(root).resolve()
    remote = _remote(remote)
    work = _work(work)
    units, skipped = scan(root)
    have, dups = remote_listing(remote)
    stored = group_remote(have, dups)
    tars, content, markers, orphans, sent, conflicts, verify = [], [], [], [], [], [], {}

    def conflict(key, why):
        conflicts.append(dict(key=key, code='E_CACHE_CONFLICT', detail=why))

    for u in ordered(units):
        if u.requires and u.requires not in units and u.requires not in stored:
            orphans.append(u.key)
            continue
        if set(u.all()) & dups or (u.kind == 'endpoints' and u.key + '.tar' in dups):
            conflicts.append(dict(key=u.key, code='E_CACHE_DUPLICATE', detail='store lists this object twice'))
            continue
        if u.kind == 'endpoints':
            m = manifest(root, u)
            if u.key + '.tar' not in have:
                tars.append((u, m))
                continue
            try:
                other = remote_manifest(remote, u.key)['content_sha256']
            except CacheStoreError as exc:
                conflicts.append(dict(key=u.key, code=exc.code, detail=exc.detail))
                continue
            if other != m['content_sha256']:
                conflict(u.key, 'stored tar has another manifest')
        elif u.kind == 'columns':
            npy, side = u.files[0], u.markers[0]
            if npy in have or side in have:
                verify[u.key] = u  # compare with the stored bytes first
            else:
                content.append(npy)
                markers.append(side)
                sent.append(u.key)
        else:
            f = u.all()[0]
            if f not in have:
                (markers if u.kind == 'attach_done' else content).append(f)
                sent.append(u.key)
            elif have[f] != (root / f).stat().st_size:
                conflict(u.key, f'stored size {have[f]} != local {(root / f).stat().st_size}')
    seconds = 0.0
    tar_bytes = {}
    if verify:  # one transfer for every stored column file needed
        work.mkdir(parents=True, exist_ok=True)
        got = Path(tempfile.mkdtemp(prefix='verify-', dir=work))
        try:
            seconds += copy_files(remote, got, [f for u in verify.values() for f in u.all() if f in have], work)
            for u in verify.values():
                npy, side = u.files[0], u.markers[0]
                local = (root / side).read_text().strip()
                if side in have:
                    if (got / side).read_text().strip() != local:
                        conflict(u.key, 'stored sha256 marker differs')
                        continue
                    if npy not in have:
                        content.append(npy)
                        sent.append(u.key)
                elif _sha_file(got / npy) == local:
                    markers.append(side)
                    sent.append(u.key)  # stored npy = ours: attach marker
                else:
                    conflict(u.key, 'stored npy without marker has other bytes')
        finally:
            shutil.rmtree(got, ignore_errors=True)
    for u, m in tars:
        up = work / f'up-{uuid.uuid4().hex[:8]}'
        try:
            (up / 'geometry').mkdir(parents=True)
            if shutil.disk_usage(up).free < u.bytes + DISK_MARGIN:
                raise CacheStoreError('E_CACHE_DISK_LOW', f'{u.bytes} bytes for {u.key}.tar')
            try:
                write_tar(root, u, m, up / (u.key + '.tar'))
            except OSError as exc:
                raise CacheStoreError(
                    'E_CACHE_DISK_FULL' if exc.errno == errno.ENOSPC else 'E_CACHE_LOCAL_IO', type(exc).__name__
                ) from None
            t0 = time.time()
            seconds += copy_files(up, remote, [u.key + '.tar'], work)
            tar_bytes[u.key] = (up / (u.key + '.tar')).stat().st_size
            say(
                event='tar_uploaded',
                key=u.key,
                bytes=tar_bytes[u.key],
                start_unix=round(t0, 2),
                end_unix=round(time.time(), 2),
            )
        except CacheStoreError as exc:
            if exc.code in ('E_CACHE_RCLONE', 'E_CACHE_RCLONE_TIMEOUT'):
                raise
            conflicts.append(dict(key=u.key, code=exc.code, detail=exc.detail))
            continue
        finally:
            shutil.rmtree(up, ignore_errors=True)
        # Two pushes of one new key at once: the store keeps one tar; ours must be the one now stored.
        try:
            now = remote_manifest(remote, u.key)['content_sha256']
        except CacheStoreError as exc:
            conflicts.append(dict(key=u.key, code=exc.code, detail=exc.detail))
            continue
        if now == m['content_sha256']:
            sent.append(u.key)
        else:
            conflict(u.key, 'another copy was stored first')
    seconds += copy_files(root, remote, content, work)
    seconds += copy_files(root, remote, markers, work)
    codes = sorted({c['code'] for c in conflicts} | ({'E_CACHE_ORPHAN'} if orphans else set()))
    row = dict(
        event='push',
        root=str(root),
        remote=redact_connections(remote, remote=True),
        units=len(units),
        units_sent=len(set(sent)),
        files_sent=len(content) + len(markers) + len(tar_bytes),
        bytes_sent=sum(tar_bytes.values()) + sum((root / f).stat().st_size for f in content + markers),
        seconds=round(seconds, 2),
        already_stored=len(units) - len(set(sent)) - len(orphans) - len(conflicts),
        conflicts=conflicts,
        orphans=orphans,
        codes=codes,
        skipped=len(skipped),
        skipped_reasons=sorted({why.split(':')[0] for _, why in skipped}),
    )
    say(**row)
    return row


@contextlib.contextmanager
def flocked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _link(src, dst):
    """No-clobber atomic publication: 'published' or 'present'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return 'published'
    except FileExistsError:
        return 'present'


def publish(u, stage, root):
    """One staged, re-checked unit into the mirror under the pipeline's own locks. Returns 'published', 'present' or
    an E_CACHE_* code (left out, nothing replaced)."""
    if u.kind == 'endpoints':
        sig = u.key.split('endpoints_', 1)[1]
        dst = root / u.key
        # Builder lock of run_gpu_state_v3.py (every build and completeness check of this signature holds it).
        with flocked(root / 'geometry' / (sig + '.lock')):
            if (dst / 'packed/complete.json').exists():
                return 'present'
            if (dst / 'packed').exists():
                return 'E_CACHE_LOCAL_PARTIAL'
            for name in ('identity.json', 'source_height.json'):
                source, target = stage / u.key / name, dst / name
                if target.exists() and target.read_bytes() != source.read_bytes():
                    return 'E_CACHE_CONFLICT'
            for name in ('identity.json', 'source_height.json'):
                source, target = stage / u.key / name, dst / name
                if _link(source, target) == 'present' and target.read_bytes() != source.read_bytes():
                    return 'E_CACHE_CONFLICT'
            os.rename(stage / u.key / 'packed', dst / 'packed')  # the whole build appears at once, complete.json inside
        return 'published'
    if u.kind == 'aerosol':
        if not (root / u.requires / 'packed/complete.json').exists():
            return 'E_CACHE_ORPHAN'
        with flocked((root / u.key).with_name((root / u.key).name + '.lock')):  # ensure_column's per-height lock
            return _link(stage / u.key, root / u.key)
    if u.kind == 'columns':
        npy, side = root / u.files[0], root / u.markers[0]
        with flocked(root / (u.key + '.lock')):  # cached_columns' per-key lock (column_cache.py)
            if npy.exists():
                return 'present'
            os.replace(stage / u.markers[0], side)
            os.replace(stage / u.files[0], npy)  # writer's order (column_cache.py)
        return 'published'
    out = 'present'
    for f in u.all():
        out = 'published' if _link(stage / f, root / f) == 'published' else out
    return out


def pull(root, keys=None, remote=None, legacy=False):
    """Download the selected store units the mirror lacks, check them, publish them. Locks are never transferred;
    legacy loose endpoints folders only with legacy=True."""
    root = Path(root).resolve()
    remote = _remote(remote)
    for k in KINDS:
        (root / k).mkdir(parents=True, exist_ok=True)
    have, dups = remote_listing(remote)
    stored = group_remote(have, dups)
    usable = {k: u for k, u in stored.items() if legacy or u.fmt != 'legacy'}
    chosen = select(usable, keys)
    local, _ = scan(root)
    need = {}
    for key, u in chosen.items():
        if key in local:
            continue
        if u.requires and u.requires not in local and u.requires not in chosen:
            need[key] = 'orphan'
            continue
        need[key] = u
    stage = root / STAGE / f'pull-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    stage.mkdir(parents=True)
    results = {k: 'E_CACHE_ORPHAN' for k, v in need.items() if v == 'orphan'}
    todo = {k: v for k, v in need.items() if v != 'orphan'}
    seconds = 0.0
    rels = []
    try:
        t0 = time.monotonic()
        for u in ordered(todo):
            if u.fmt != 'tar':
                continue
            try:
                stream_tar(remote, u.key, stage)
            except CacheStoreError as exc:
                results[u.key] = exc.code
                shutil.rmtree(stage / u.key, ignore_errors=True)
            except OSError as exc:
                results[u.key] = 'E_CACHE_DISK_FULL' if exc.errno == errno.ENOSPC else 'E_CACHE_LOCAL_IO'
                shutil.rmtree(stage / u.key, ignore_errors=True)
        seconds += time.monotonic() - t0
        rels = [f for u in ordered(todo) if u.fmt != 'tar' for f in u.all()]
        try:
            seconds += copy_files(remote, stage, rels, root / STAGE)
        except OSError as exc:
            raise CacheStoreError(
                'E_CACHE_DISK_FULL' if exc.errno == errno.ENOSPC else 'E_CACHE_LOCAL_IO', type(exc).__name__
            ) from None
        staged, _ = scan(stage)
        for u in ordered(todo):
            if u.key in results:
                continue
            if u.key not in staged:
                results[u.key] = 'E_CACHE_STAGED_INCOMPLETE'
                continue
            try:
                results[u.key] = publish(u, stage, root)
            except OSError as exc:
                results[u.key] = 'E_CACHE_DISK_FULL' if exc.errno == errno.ENOSPC else 'E_CACHE_LOCAL_IO'
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        with contextlib.suppress(OSError):
            (root / STAGE).rmdir()
    codes = sorted({v for v in results.values() if v.startswith('E_')})
    row = dict(
        event='pull',
        root=str(root),
        remote=redact_connections(remote, remote=True),
        selected=len(chosen),
        already_local=len(chosen) - len(need),
        published=sum(v == 'published' for v in results.values()),
        present=sum(v == 'present' for v in results.values()),
        files=len(rels) + sum(u.fmt == 'tar' for u in todo.values()),
        bytes=sum(u.bytes for u in todo.values()),
        seconds=round(seconds, 2),
        codes=codes,
        failed={k: v for k, v in results.items() if v.startswith('E_')},
        legacy_skipped=sorted(k for k, u in stored.items() if u.fmt == 'legacy' and not legacy),
    )
    say(**row)
    return row


def _anonymous(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ''  # no Mac account names in a rig input
    return info


def pack_for_rig(keys, out, local=None, remote=None):
    """R2 cache input for a rig bake (extract_to: work): a plain tar of cache/<unit files> (the local layout). From a
    local mirror, else the selected units are pulled into a temporary mirror next to out first. A failure leaves
    neither the temporary mirror nor a partial tar, so the same command can simply be run again."""
    out = Path(out).resolve()
    tmp_mirror = None
    part = out.with_name(out.name + '.part')
    try:
        if local is None:
            tmp_mirror = out.with_name(out.name + '.mirror')
            try:
                tmp_mirror.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                tmp_mirror = None
                raise CacheStoreError('E_CACHE_LOCAL_ROOT', f'{out.name}.mirror: {type(exc).__name__}') from None
            row = pull(tmp_mirror, keys, remote)
            if row['codes']:
                raise CacheStoreError(row['codes'][0], 'pull for the rig tar failed: ' + json.dumps(row['failed']))
            local = tmp_mirror
        units, _ = scan(Path(local))
        chosen = select(units, keys)
        for u in chosen.values():
            if u.requires and u.requires not in chosen:
                raise CacheStoreError('E_CACHE_ORPHAN', u.key + ' without ' + u.requires)
        try:
            with tarfile.open(part, 'w', format=tarfile.PAX_FORMAT) as t:
                for u in ordered(chosen):
                    for f in u.all():
                        t.add(Path(local) / f, arcname='cache/' + f, recursive=False, filter=_anonymous)
            os.replace(part, out)
        except OSError as exc:
            raise CacheStoreError(
                'E_CACHE_DISK_FULL' if exc.errno == errno.ENOSPC else 'E_CACHE_LOCAL_IO', type(exc).__name__
            ) from None
    finally:
        part.unlink(missing_ok=True)
        if tmp_mirror is not None:
            shutil.rmtree(tmp_mirror, ignore_errors=True)
    row = dict(
        event='pack_for_rig',
        out=str(out),
        units=len(chosen),
        files=sum(len(u.all()) for u in chosen.values()),
        bytes=out.stat().st_size,
        sha256=_sha_file(out),
        keys=sorted(chosen),
    )
    say(**{k: v for k, v in row.items() if k != 'keys'})
    return row


def list_store(remote=None):
    have, dups = remote_listing(remote)
    units = group_remote(have, dups)
    rows = [u.row() for u in ordered(units)]
    used = {f for u in units.values() for f in u.all()}
    legacy_shadowed = sorted(f for f in have if LEGACY.fullmatch(f) and f not in used)  # loose files a tar supersedes
    loose = sorted(set(have) - used - set(legacy_shadowed))
    return dict(
        units=rows,
        bytes=sum(have.values()),
        files=len(have),
        duplicates=sorted(dups),
        legacy_superseded=legacy_shadowed,
        incomplete_or_unknown=loose,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('list')
    s.add_argument('--json', action='store_true')
    s = sub.add_parser('push')
    s.add_argument('local_root', type=Path)
    s.add_argument('--report', type=Path)
    s = sub.add_parser('pull')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--all', action='store_true')
    g.add_argument('--keys', nargs='+')
    s.add_argument('--legacy', action='store_true', help='also read loose endpoints folders of the first fill')
    s.add_argument('local_root', type=Path, nargs='?')
    s.add_argument('--report', type=Path)
    s = sub.add_parser('pack-for-rig')
    s.add_argument('--local', type=Path)
    s.add_argument('--all', action='store_true')
    s.add_argument('items', nargs='+', help='keys..., then out.tar')
    a = p.parse_args(argv)
    if a.cmd == 'pull' and a.local_root is None:  # `pull --keys K ... <local_root>`: --keys took the root too
        if not a.keys or len(a.keys) < 2:
            p.error('pull needs <local_root>')
        a.local_root = Path(a.keys.pop())
    try:
        if a.cmd == 'list':
            row = list_store()
            if a.json:
                print(json.dumps(row, indent=1))
            else:
                for r in row['units']:
                    print(f"{r['bytes']:>12d}  {r['files']:>3d}  {r['kind']:<17s} {r['key']}")
                print(
                    f"total {row['bytes']} bytes, {row['files']} files, {len(row['units'])} units, {len(row['incomplete_or_unknown'])} loose, "
                    f"{len(row['legacy_superseded'])} superseded legacy, {len(row['duplicates'])} duplicated"
                )
            return 1 if row['duplicates'] else 0
        if a.cmd == 'push':
            row = push(a.local_root)
        elif a.cmd == 'pull':
            row = pull(a.local_root, None if a.all else a.keys, legacy=a.legacy)
        else:
            *keys, out = a.items
            if not a.all and not keys:
                raise CacheStoreError('E_CACHE_ARGS', 'pack-for-rig needs keys or --all')
            row = pack_for_rig(None if a.all else keys, out, a.local)
        if getattr(a, 'report', None):
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(row, indent=1) + '\n')
        return 1 if row.get('codes') else 0
    except CacheStoreError as exc:
        code, detail = exc.code, clean(exc.detail)
    except Exception as exc:  # anything else still leaves a stable code
        code, detail = 'E_CACHE_INTERNAL', clean(f'{type(exc).__name__}: {exc}')
    say(event='error', cmd=a.cmd, code=code, detail=detail)
    print(f'{code}: {detail}', file=sys.stderr, flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
