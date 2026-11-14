#!/usr/bin/env bash
# Build into an empty directory on Debian or Ubuntu x86_64.
set -euo pipefail
[[ $(uname -m) == x86_64 ]] || { echo 'E_ARCH: x86_64 required' >&2; exit 1; }
[[ $# == 1 && -d $1 ]] || { echo 'Usage: bash build_libradtran.sh <empty directory>' >&2; exit 2; }
work=$(cd "$1" && pwd)
[[ -z $(ls -A "$work") ]] || { echo 'E_NOT_EMPTY: an empty directory is required' >&2; exit 2; }
SUDO=()
if [[ $(id -u) != 0 ]]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential gfortran libnetcdf-dev libgsl-dev flex bison curl ca-certificates python3 netcdf-bin
cd "$work"
curl -fL --retry 3 -o libRadtran-2.0.6.tar.gz https://www.libradtran.org/download/libRadtran-2.0.6.tar.gz || \
  { echo 'E_LRT_DOWNLOAD: libRadtran-2.0.6.tar.gz could not be downloaded' >&2; exit 4; }
printf '%s  %s\n' 64930cc40b6e4a37aa220520974d330fc1563796f466a649b2238131f2d69840 \
  libRadtran-2.0.6.tar.gz | sha256sum -c - || { echo 'E_LRT_SHA: unexpected libRadtran-2.0.6.tar.gz' >&2; exit 5; }
tar -xzf libRadtran-2.0.6.tar.gz
cd libRadtran-2.0.6
# Regenerate dependency paths.
find . -name .depend -type f -delete
CC='gcc -std=gnu11' CFLAGS='-O2' FFLAGS='-O2 -fallow-argument-mismatch' ./configure --with-disable-opengl
grep -Eq '^FOUND_MYSTIC=.yes.' config.log || { echo 'E_NO_MYSTIC: inspect config.log' >&2; exit 3; }
make -j"$(nproc)" || make -j2
test -x bin/uvspec
# Check a simple 1D MYSTIC run.
mkdir smoke
cd smoke
cat > smoke.inp <<'EOF'
data_files_path ../data/
atmosphere_file ../data/atmmod/afglus.dat
wavelength 550 550
rte_solver mystic
mc_photons 10000
sza 30
output_user lambda edir edn eup
quiet
EOF
../bin/uvspec < smoke.inp > smoke.out
python3 - <<'PY'
import math
from pathlib import Path
rows = [line.split() for line in Path('smoke.out').read_text().splitlines() if line.strip()]
if not rows or any(len(row) != 4 for row in rows):
    raise SystemExit('E_SMOKE_SHAPE')
values = [[float(x) for x in row] for row in rows]
if not all(math.isfinite(x) for row in values for x in row) or not any(row[1] > 0 for row in values):
    raise SystemExit('E_SMOKE_VALUE')
print('MYSTIC_SMOKE_OK')
PY
