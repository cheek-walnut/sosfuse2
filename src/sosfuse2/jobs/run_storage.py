"""On-disk run outputs, source reuse and archive verification."""

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

from ..atmosphere.components import check_current, check_optics
from ..foundation import paths


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def update_metadata(path, changes):
    """Extend the latest record, including the peak RSS and elapsed time updated by the packer."""
    meta = json.loads(path.read_text())
    meta.update(changes)
    atomic_json(path, meta)


class FirstOrderConflict(RuntimeError):
    """The loose and archived first-order files have different digests."""


def verify_pack(path, signature, *, export_first_order=None, first_order_sha=None, table_sha=None,
                loose_table=None, restore_first=None):
    # Read every archived member through zstd; verify both CRC/content and the identity.
    # tarfile's stream mode stops at the end-of-archive marker and leaves the record padding (10240 bytes measured) unread;
    # closing the pipe then can kill a zstd still writing it (SIGPIPE, nonzero exit) when the pipe buffer is small, which
    # can reject a valid pack. The rest of the stream
    # is drained before close, and a real decode failure reports zstd's exit code and stderr.
    import tempfile

    from ..outputs.radiance_table import exports_first_order, read_table

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(['zstd', '-q', '-dc', str(path)], stdout=subprocess.PIPE, stderr=err)
        failed = False
        tar_error = None
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                names = []
                meta = None
                for member in archive:
                    if not member.isfile():
                        raise RuntimeError('E_PACK_VERIFY: unexpected member')
                    names.append(member.name)
                    f = archive.extractfile(member)
                    if member.name == 'meta.json':
                        meta = json.load(f)
                    elif member.name == 'table.npz' and table_sha is not None:
                        if hashlib.file_digest(f, 'sha256').hexdigest() != table_sha:
                            raise RuntimeError('E_PACK_VERIFY: archived radiance table differs')
                    elif member.name == 'table.npz' and loose_table is not None:
                        import numpy as np

                        with tempfile.TemporaryFile() as archived, loose_table.open('rb') as loose:
                            shutil.copyfileobj(f, archived)
                            archived.seek(0)
                            if hashlib.file_digest(archived, 'sha256').digest() != hashlib.file_digest(loose, 'sha256').digest():
                                archived.seek(0)
                                loose.seek(0)
                                aa, ar = read_table(archived)
                                ba, br = read_table(loose)
                                if (any(a.dtype != b.dtype or not np.array_equal(a, b) for a, b in zip(aa, ba))
                                        or not np.allclose(ar, br, rtol=1e-5, atol=0)):
                                    raise RuntimeError('E_PACK_VERIFY: archived radiance table differs')
                    elif member.name == 'first_order.npz' and restore_first is not None:
                        shutil.copyfileobj(f, restore_first)
                    elif member.name == 'first_order.npz' and first_order_sha is not None:
                        if hashlib.file_digest(f, 'sha256').hexdigest() != first_order_sha:
                            raise FirstOrderConflict('E_PACK_VERIFY: first-order table differs')
                    else:
                        while f.read(1024 * 1024):
                            pass
                expected = ['table.npz', 'meta.json', 'preview_m03.png', 'preview_m06.png']
                exported = isinstance(meta, dict) and exports_first_order(meta)
                if export_first_order is not None and export_first_order != exported:
                    raise RuntimeError('E_PACK_VERIFY: first-order setting differs')
                if exported:
                    expected.append('first_order.npz')
                    first = meta.get('first_order')
                    if not isinstance(first, dict) or first.get('file') != 'first_order.npz':
                        raise RuntimeError('E_PACK_VERIFY: first-order metadata missing')
                if (
                    sorted(names) != sorted(expected)
                    or not isinstance(meta, dict)
                    or meta.get('complete') is not True
                    or meta.get('signature') != signature
                ):
                    raise RuntimeError('E_PACK_VERIFY: members or identity')
        except (tarfile.TarError, EOFError) as exc:
            tar_error = exc
            raise
        except BaseException:
            failed = True
            raise
        finally:
            try:
                while proc.stdout.read(1024 * 1024):
                    pass
            except OSError:
                pass
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0 and not failed:
                err.seek(0)
                detail = err.read().decode(errors='replace').strip()[-300:]
                message = f'E_PACK_VERIFY: zstd decode failed (exit {rc}{": " + detail if detail else ""})'
                if tar_error is not None:
                    message += f'; {type(tar_error).__name__}: {tar_error}'
                raise RuntimeError(message) from tar_error


def output_identity_message(path, reason):
    return (
        f'E_OUTPUT_IDENTITY: {path}: {reason}. '
        'If this output is from an earlier code version, use a separate output root or inspect and move the old output.'
    )


def enable_spectral_previews(module):
    import numpy as np

    original = module.previews

    def previews(out):
        from ..atmosphere.spectral import check_meta_contract

        meta = json.loads((out / 'meta.json').read_text())
        check_meta_contract(meta)
        with np.load(out / 'table.npz') as f:
            table = f['radiance']
            wl = f['wavelength_nm']
            hs = f['solar_elevation_deg']
        if len(wl) in (1, 15):
            return original(out)
        weights = module.preview_weights(
            wl, module.preview_band_widths(meta, wl), np.load(paths.ASSETS / 'rgb_weights.npy')
        )
        rgb = table @ weights
        funcs = module.render_functions()
        for h in (-3, -6):
            folded = rgb[np.flatnonzero(hs == h)[0]]
            pano = np.concatenate([folded[:, ::-1], folded], axis=1)
            projected = funcs['project'](pano, funcs['projection'](960, 540, 100, 0))
            k = funcs['central'](projected)[1] if (projected @ funcs['Y']).max() > 0 else 1.0
            funcs['save_png'](out / f'preview_m{abs(h):02d}.png', funcs['tone'](projected, k))

    module.previews = previews


@contextlib.contextmanager
def output_lock(out):
    """Hold the output's exclusive worker lock through all writes and verification."""
    with (out / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('E_OUTPUT_BUSY: out_name already owned by another worker')
        yield


@contextlib.contextmanager
def source_lock(src_out):
    """Pin the source arrays, metadata and anchor table together for one derived run."""
    path = src_out / '.worker.lock'
    try:
        lock = path.open('a')
    except FileNotFoundError:
        raise RuntimeError('E_EXTRA_SOURCE: source job meta missing')
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        try:
            lock = path.open('r')
        except FileNotFoundError:
            # Read-only source folder with no existing worker lock.
            yield
            return
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('E_EXTRA_SOURCE: source job is owned by a running worker (.worker.lock)')
        yield


def check_output(out, signature, optics):
    """Allow an interrupted or completed output with the same optical state and signature."""
    meta_path = out / 'meta.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        check_optics(previous.get('optics_state'), optics)
        if previous.get('signature') != signature:
            raise RuntimeError(output_identity_message(out, f"signature differs (stored format={previous.get('contract')!r})"))


def read_source_metadata(src_out, src, optics, expected):
    """Verify the source job before solver construction, under its shared lock."""
    try:
        meta = json.loads((src_out / 'meta.json').read_text())
    except (OSError, ValueError):
        raise RuntimeError('E_EXTRA_SOURCE: source job meta missing')
    if not src.is_file():
        raise RuntimeError('E_EXTRA_SOURCE: source.npz missing')
    check_current(meta.get('physics'), what='source job')
    if meta.get('density') is None:
        raise RuntimeError('E_EXTRA_SOURCE: density profile missing')
    check_optics(meta.get('optics_state'), optics)
    bad = [key for key, value in expected.items() if meta.get(key) != value]
    if bad:
        raise RuntimeError('E_EXTRA_SOURCE: source meta differs: ' + ','.join(bad))
    return meta


def invalidate_completed(out):
    """Keep the signature while marking a rerun incomplete before replacing any arrays."""
    meta_path = out / 'meta.json'
    if meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        if previous.get('complete'):
            atomic_json(meta_path, dict(previous, complete=False))


def write_source(prepare, fields, h, chi, wl, *, physical, optics, contract):
    """Store CPU float32 source chunks (flattened height/zenith cell, expansion row, wavelength).
    The final axis contains physical wavelengths in nm, without padding. h (km), chi
    (radians) and wl are 1-D host coordinates, normally float64. Strided chunks are
    accepted and joined along wavelength. All inputs are read without in-place changes.
    """
    import numpy as np

    source = prepare / 'source.npz'
    tmp = source.with_suffix('.npz.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez(
                f,
                source=np.concatenate(fields, axis=-1),
                h=h,
                chi=chi,
                wavelength_nm=wl,
                physics=np.array(json.dumps(physical, sort_keys=True)),
                optics_state=np.array(json.dumps(optics, sort_keys=True)),
                source_contract=np.array(json.dumps(contract, sort_keys=True, allow_nan=False)),
            )
        os.replace(tmp, source)
    finally:
        tmp.unlink(missing_ok=True)


def write_moments(prepare, moments, h, chi, wl, order):
    """Store CPU float32 moment chunks (cell, moment slot, wavelength), accepting strided views."""
    import numpy as np

    np.savez(
        prepare / 'moment.npz',
        moment=np.concatenate(moments, axis=-1),
        h=h,
        chi=chi,
        wavelength_nm=wl,
        L=np.int32(order),
        layout=np.array(
            '(cell=ih*n_chi+ic, slot, channel); slot [0:4] Rayleigh coherency, [4:4+K] scalar radiance SH k=l(l+1)/2+m (m>=0, cos m phi), frame up=radial ex=sun-ward horizontal'
        ),
    )


def write_table(out, table, wl, hs, el, az, *, first_tables=None):
    """Write the README radiance table in W m^-2 sr^-1 nm^-1.

    Axes are solar elevation, view elevation, relative azimuth (all degrees), and
    wavelength (nm). The CPU float32 table normally has contiguous wavelength lanes.
    NumPy strides are accepted for every array. Coordinates are saved as float32.
    The writer reads the arrays without modifying them. Optional first-order chunks
    have the same axes and units and cover successive slices of the wavelength axis.
    """
    import numpy as np
    from ..outputs.radiance_table import save_first_order

    with (out / 'table.npz.tmp').open('wb') as f:
        np.savez(
            f,
            radiance=table,
            wavelength_nm=wl.astype('f4'),
            solar_elevation_deg=hs.astype('f4'),
            view_elevation_deg=el.astype('f4'),
            relative_azimuth_deg=az.astype('f4'),
            units=np.array('W m^-2 sr^-1 nm^-1'),
        )
    os.replace(out / 'table.npz.tmp', out / 'table.npz')
    return save_first_order(out, first_tables, wl, hs, el, az) if first_tables is not None else {}


def write_run_arrays(out, prepare, table, wl, hs, el, az, *, h, chi, fields, moments, order,
                     physical, optics, contract, first_tables=None):
    """Publish one solve's host arrays under its output lock. See write_source and write_table for layouts."""
    invalidate_completed(out)
    write_source(prepare, fields, h, chi, wl, physical=physical, optics=optics, contract=contract)
    if moments:
        write_moments(prepare, moments, h, chi, wl, order)
    return write_table(out, table, wl, hs, el, az, first_tables=first_tables)


def pack_output(out, pack_dir, started, signature, *, verify, reuse_previews=False):
    """Publish and check the archive while the caller still owns the output."""
    from ..outputs import pack_state

    if reuse_previews:
        pack = pack_state.pack_state(out, pack_dir, 'zstd', started=started, reuse_previews=True)
    else:
        enable_spectral_previews(pack_state)
        pack = pack_state.pack_state(out, pack_dir, 'zstd', started=started)
    verify(pack, signature)
    return pack


def check_anchor_rows(src_out, table, hs):
    """Compare the two shared elevation rows while the source lock is held."""
    import numpy as np

    with np.load(src_out / 'table.npz') as f:
        ref = f['radiance']
        ref_hs = f['solar_elevation_deg']
    anchors = {}
    for h in (-3.0, -6.0):
        a_row = table[int(np.flatnonzero(hs == h)[0])]
        b_row = ref[int(np.flatnonzero(ref_hs == h)[0])]
        anchors[str(h)] = dict(
            max_rel=float(np.max(np.abs(a_row - b_row)) / max(float(np.max(np.abs(b_row))), 1e-30))
        )
    for row in anchors.values():
        row['within_tol'] = row['max_rel'] <= 1e-6
    if max(v['max_rel'] for v in anchors.values()) > 1e-6:
        raise RuntimeError('E_EXTRA_ANCHOR: anchor rows differ from source table by >1e-6')
    return anchors
