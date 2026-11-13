"""IPRT (libRadtran intercomparison, phase E) layer optical-depth inputs, fetched on first use.

These two small text files are not redistributed with the package (the IPRT wiki
states no per-file terms; its footer licence is CC BY-SA 3.0 for wiki content). They are downloaded from the
original server into the work root and verified against pinned SHA-256 digests, so the derived US1976 + IPRT
ozone-shape profile (generate_density_profile.py) is reproduced exactly from the original inputs.

A failed download is retried (3 attempts, short backoff) and then reported as E_IPRT_DOWNLOAD with the file, the URL and the cause.
"""

import hashlib
import http.client
import os
import time
import urllib.request
from pathlib import Path

from . import paths

BASE = 'https://www.libradtran.org/iprt/lib/exe/fetch.php?media=intercomparisons:phase_e:'
FILES = {  # name -> sha256 of the original server file
    'tau_rayleigh_450nm_usstd.dat': '5c7550337bc1bacbb3b935d96c5cdc45ba444a4dc644b5398f6f8006d22a23d4',
    'tau_absorption_320nm_usstd.dat': '2cdde3d5280e8bdd1798b2b3974959725f3311be5be6628330cfdf7b46e4ec43',
}
DOWNLOAD_ATTEMPTS = 3
RETRY_SECONDS = 1.0  # waits 1 s, 2 s between attempts
CITATION = (
    'Emde, C. et al., IPRT polarized radiative transfer model intercomparison project, phase E inputs '
    '(US standard atmosphere layer optical depths), https://www.libradtran.org/iprt/'
)


def cache_dir():
    d = paths.work_root() / 'iprt'
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(name, offline=None):
    """Return the local path of an IPRT input, downloading and verifying it once.
    offline=True (or SOSFUSE2_OFFLINE=1) refuses to download: E_IPRT_OFFLINE with the URL to fetch by hand."""
    if name not in FILES:
        raise KeyError(f'E_IPRT_UNKNOWN: {name}')
    want = FILES[name]
    dst = cache_dir() / name
    if dst.is_file() and sha256(dst) == want:
        return dst
    if offline is None:
        offline = os.environ.get('SOSFUSE2_OFFLINE', '') not in ('', '0')
    url = BASE + name
    if offline:
        raise RuntimeError(
            f'E_IPRT_OFFLINE: {name} missing from {dst.parent}; download {url} and place it there (sha256 {want})'
        )
    tmp = dst.with_suffix(dst.suffix + f'.{os.getpid()}.tmp')
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            break
        except (
            OSError,
            http.client.HTTPException,
        ) as e:  # URLError, HTTPError, TimeoutError, ConnectionError, IncompleteRead
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f'E_IPRT_DOWNLOAD: {name} from {url}: {type(e).__name__}: {e} ({attempt} attempts)'
                ) from e
            time.sleep(RETRY_SECONDS * 2 ** (attempt - 1))
    tmp.write_bytes(data)
    got = sha256(tmp)
    if got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'E_IPRT_SHA256: {name} from {url}: got {got}, expected {want}')
    os.replace(tmp, dst)
    return dst


def inputs(offline=None):
    """Both generator inputs as {name: Path}, plus their digests for provenance."""
    got = {n: fetch(n, offline) for n in FILES}
    return got, dict(FILES)
