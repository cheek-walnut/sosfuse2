#!/usr/bin/env python3
"""D0 appreciation renders. Read-only inputs; writes only results/sos_render/.

Run: .venv-mac-mi/bin/python -B scripts/sos_render.py
No renderer, GPU, package installation, or existing-file modification.
"""
import os
import sys
sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import ast
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
from machine_config import machine_path
import shutil
import struct
import subprocess
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sos_render'
ES = (2, 0, -3, -6, -9, -12)
Y = np.array([.2126, .7152, .0722], np.float32)
NIT = 72983.
ADAPT = .70
M2020 = np.array([[.627403896, .329283038, .043313066],
                 [.069097289, .919540395, .011362316],
                 [.016391439, .088013308, .895595253]], np.float32)
VIEWS = {'sun': (1920, 1080, 100, 0), 'phone': (1080, 2340, 60, 0),
         'anti': (1920, 1080, 100, 180)}
MODES = {'a': '카메라 자동노출', 'b': '눈에 보이는 밝기', 'c': '푸르키네 혼합'}


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def reuse(filename, names, scope=None):
    """Reuse pure existing functions without importing renderer-dependent modules."""
    scope = dict(globals()) if scope is None else scope
    tree = ast.parse((ROOT / 'scripts' / filename).read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == set(names)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(ROOT / 'scripts' / filename), 'exec'), scope)
    return scope


PA = reuse('purkinje_analyze.py', ('weights', 'mesopic'),
           {**globals(), 'EDGES': np.linspace(360, 830, 16),
            'integral_samples': lambda lo, hi: np.unique(np.r_[lo, np.arange(math.ceil(lo), hi), hi])})
HDR = reuse('hdr_export.py', ('pq', 'unpq', 'png16'))
APP = reuse('sasktran_probe.py', ('app_lut', 'app_pano', 'pano_from_grid', 'pano_coords'))


def tag(e):
    return ('p' if e >= 0 else 'm') + f'{abs(e):02d}'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def film(x):
    """Hable rational shoulder, asymptotic maximum .98; zero stays zero."""
    a, b, c, d, e, f = .15, .50, .10, .20, .02, .30
    return .98 / (1 - e / f) * (
        (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f)


def tone(rgb, k, headroom=1.):
    # One scalar for all three channels: no white balance, hue rotation, or saturation boost.
    x = np.maximum(rgb * (k / headroom), 0)
    maximum = x.max(-1)
    scale = np.divide(film(maximum), maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return np.maximum(x * scale[..., None] * headroom, 0)


def encode(rgb, seed=17):
    v = np.where(rgb <= .0031308, 12.92 * rgb,
                 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    # Shared-channel, zero-mean dither immediately before 8-bit quantization.
    rng = np.random.default_rng(seed)
    dither = rng.uniform(-.50, .50, (*rgb.shape[:-1], 1)).astype(np.float32)
    return np.uint8(np.clip(np.rint(v * 255 + dither), 0, 255))


def save_png(path, rgb, seed=17):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add(b'sRGB', b'\x00')
    Image.fromarray(encode(rgb, seed)).save(path, compress_level=9, pnginfo=info)


def resize_linear(rgb, width):
    height = round(rgb.shape[0] * width / rgb.shape[1])
    return np.stack([np.asarray(Image.fromarray(rgb[..., c], mode='F').resize(
        (width, height), Image.Resampling.LANCZOS)) for c in range(3)], -1).clip(0, 1)


def projection(w, h, hfov, azimuth):
    """Gnomonic pinhole; horizontal field of view at the image centre; horizon .8H.

    Uses the centre/right/up construction in lut_mockup.desk_dirs, with the
    specified field of view and analytically chosen camera pitch.
    """
    focal = w / (2 * math.tan(math.radians(hfov / 2)))
    pitch = math.atan(.30 * h / focal)
    a = math.radians(azimuth)
    centre = np.array([math.cos(pitch) * math.sin(a), math.cos(pitch) * math.cos(a), math.sin(pitch)])
    right = np.array([math.cos(a), -math.sin(a), 0.])
    up = np.array([-math.sin(pitch) * math.sin(a), -math.sin(pitch) * math.cos(a), math.cos(pitch)])
    xx, yy = np.meshgrid((np.arange(w, dtype=np.float32) + .5 - w / 2) / focal,
                         (h / 2 - np.arange(h, dtype=np.float32) - .5) / focal)
    p = centre.astype(np.float32) + xx[..., None] * right.astype(np.float32) + yy[..., None] * up.astype(np.float32)
    az = np.degrees(np.arctan2(p[..., 0], p[..., 1]))
    alt = np.degrees(np.arctan2(p[..., 2], np.hypot(p[..., 0], p[..., 1])))
    return az, alt


def cubic_weights(t):
    return (-.5*t+t*t-.5*t**3, 1-2.5*t*t+1.5*t**3,
            .5*t+2*t*t-1.5*t**3, -.5*t*t+.5*t**3)


def sample(pano, az, alt):
    """Catmull-Rom bicubic, periodic azimuth, clamped altitude; bound overshoots.

    Supports the original 100x360 SOS and app_pano's 180x720 grids.
    """
    nr, nc, channels = pano.shape
    out = np.empty((*az.shape, channels), np.float32)
    for start in range(0, az.shape[0], 64):
        sl = slice(start, start + 64)
        rr = np.clip((90 - np.maximum(alt[sl], 0)) * nr / 90 - .5, 0, nr - 1)
        cc = (az[sl] + 180) * nc / 360 - .5
        ri, ci = np.floor(rr).astype(np.int32), np.floor(cc).astype(np.int32)
        wr, wc = cubic_weights(rr - ri), cubic_weights(cc - ci)
        val = np.zeros((*rr.shape, channels), np.float32)
        lower = np.full_like(val, np.inf)
        upper = np.full_like(val, -np.inf)
        for ir in range(4):
            r = np.clip(ri + ir - 1, 0, nr - 1)
            for ic in range(4):
                q = pano[r, (ci + ic - 1) % nc]
                val += q * (wr[ir] * wc[ic])[..., None]
                if ir in (1, 2) and ic in (1, 2):
                    np.minimum(lower, q, out=lower)
                    np.maximum(upper, q, out=upper)
        out[sl] = np.clip(val, lower, upper)
    return out


def project(pano, coords):
    az, alt = coords
    rgb = sample(pano, az, alt)
    # Identical horizon colour on each side of alt=0; quickly fading plain ground.
    below = np.minimum(alt, 0)
    fade = .001 + .999 * np.exp(below / 1.)
    rgb *= fade[..., None]
    return rgb


def central(rgb):
    h, w = rgb.shape[:2]
    region = rgb[round(.25*h):round(.55*h), round(.40*w):round(.60*w)]
    lum = region @ Y
    key = float(np.median(lum))
    representative = np.median(region.reshape(-1, 3), axis=0)
    representative *= key / max(float(representative @ Y), 1e-30)
    lo, hi = 0., 10. / key
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(tone(representative, mid) @ Y) < .18:
            lo = mid
        else:
            hi = mid
    return key, (lo + hi) / 2


def purkinje(rgb, spec, cie, e):
    if e >= -6:
        return rgb.copy(), {'active': False, 'reason': '태양고도 −6° 미만에서만 적용'}
    wp, ws = PA['weights'](cie, 'V_photopic'), PA['weights'](cie, 'V_scotopic')
    lp_spec, ls_spec = 683 * (spec @ wp), 1700 * (spec @ ws)
    ratio = np.divide(ls_spec, lp_spec, out=np.zeros_like(lp_spec), where=lp_spec > 0)
    lp = rgb @ Y
    lm, m = PA['mesopic'](lp, lp * ratio)
    mixed = m[..., None] * rgb + (1-m)[..., None] * lm[..., None]
    mixed = np.where((lp >= 5)[..., None], rgb, mixed).astype(np.float32)
    return mixed, {'active': True, 'median_m': float(np.median(m)),
                   'median_SP': float(np.median(ratio)), 'rod_colour': 'D65 neutral'}


def run_tool(args, logfile):
    log('HDR ' + Path(str(args[0])).name)
    with logfile.open('a') as f:
        f.write('\n$ ' + ' '.join(map(str, args)) + '\n')
        f.flush()
        process = subprocess.Popen(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT,
                                   cwd=OUT, env=os.environ.copy())
        while True:
            try:
                code = process.wait(timeout=120)
                break
            except subprocess.TimeoutExpired:
                log('HDR 계산 진행 중: ' + Path(str(args[0])).name)
        if code:
            raise RuntimeError(f'{args[0]} exit={code}; {logfile.name}')


def read_png16(path):
    """Read 16-bit RGB PNG without requiring an EXR/renderer package."""
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos, payload = 8, bytearray()
    while pos < len(data):
        count = struct.unpack_from('>I', data, pos)[0]
        name, block = data[pos+4:pos+8], data[pos+8:pos+8+count]
        if name == b'IHDR':
            w, h, bits, colour, _, _, interlace = struct.unpack('>IIBBBBB', block)
            assert (bits, colour, interlace) == (16, 2, 0)
        if name == b'IDAT':
            payload.extend(block)
        pos += count + 12
    raw = np.frombuffer(zlib.decompress(payload), np.uint8).reshape(h, w*6+1)
    reconstructed = np.empty((h, w*6), np.uint8)
    for row in range(h):
        kind, values = raw[row, 0], raw[row, 1:].copy()
        up = reconstructed[row-1] if row else np.zeros(w*6, np.uint8)
        if kind == 1:
            values = np.cumsum(values.reshape(w, 6), axis=0, dtype=np.uint64).astype(np.uint8).ravel()
        elif kind == 2:
            values = (values.astype(np.uint16) + up).astype(np.uint8)
        elif kind in (3, 4):
            for col in range(w*6):
                a = int(values[col-6]) if col >= 6 else 0
                b = int(up[col])
                c = int(up[col-6]) if col >= 6 else 0
                if kind == 3:
                    predict = (a+b)//2
                else:
                    p = a+b-c
                    pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                    predict = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                values[col] = (int(values[col]) + predict) % 256
        else:
            assert kind == 0
        reconstructed[row] = values
    return reconstructed.tobytes(), (h, w, 3)


def export_hdr(rgb, k, stem):
    folder = OUT / 'hdr'
    folder.mkdir(exist_ok=True)
    work = OUT / '_tmp' / stem
    work.mkdir(parents=True, exist_ok=True)
    logfile = folder / (stem + '.log')
    h, w = rgb.shape[:2]
    # Same exposure b; extend film shoulder to 980 nit, SDR reference white 200 nit.
    hdrnit = tone(rgb, k, headroom=5.) * 200
    bt2020 = hdrnit @ M2020.T
    pq = HDR['pq'](bt2020)
    pqfile = work / 'pq16.png'
    HDR['png16'](pqfile, np.uint16(np.rint(pq * 65535)))
    details = {'shoulder_max_nit': 980, 'SDR_reference_white_nit': 200,
               'max_luminance_nit': float((hdrnit @ Y).max()), 'files': [], 'errors': []}
    try:
        if shutil.which('avifenc') and shutil.which('avifdec'):
            target = folder / (stem + '.avif')
            run_tool(['avifenc', '--cicp', '9/16/9', '--depth', '10', '-y', '444',
                      '-s', '4', '-q', '90', '-j', '4', pqfile, target], logfile)
            back = work / 'avif_back.png'
            run_tool(['avifdec', '--depth', '16', '-j', '4', target, back], logfile)
            raw, shape = read_png16(back)
            codes = np.frombuffer(raw, '>u2').reshape(shape).astype(np.float32)/65535
            decoded_nit = HDR['unpq'](codes)
            # Absolute PQ error, including the darkest plates; not a fragile relative-to-zero check.
            details['avif_decoded_mean_abs_nit_error'] = float(np.abs(decoded_nit-bt2020).mean())
            details['avif_decoded_shape'] = list(shape)
            assert shape == hdrnit.shape
            details['files'].append(str(target.relative_to(OUT)))
        else:
            details['errors'].append('AVIF 생략: avifenc/avifdec 도구 없음')
    except Exception as exc:
        details['errors'].append('AVIF 실패: ' + str(exc))
    try:
        if shutil.which('ultrahdr_app'):
            # Installed v2.0.2 bounds its CPU workers to four (jpegr.cpp).
            q = np.uint32(np.rint(pq * 1023))
            packed = q[..., 0] | (q[..., 1] << 10) | (q[..., 2] << 20) | np.uint32(3 << 30)
            rawhdr, rawsdr = work / 'hdr.raw', work / 'sdr.raw'
            packed.astype('<u4').tofile(rawhdr)
            rgba = np.empty((h, w, 4), np.uint8)
            rgba[..., :3] = np.asarray(Image.open(OUT/'full'/(stem+'.png')))
            rgba[..., 3] = 255
            rgba.tofile(rawsdr)
            target = folder / (stem + '_ultrahdr.jpg')
            run_tool(['ultrahdr_app', '-m', '0', '-p', rawhdr, '-y', rawsdr,
                      '-a', '5', '-b', '3', '-w', w, '-h', h, '-C', '2', '-c', '0',
                      '-t', '2', '-R', '1', '-q', '95', '-Q', '95', '-u', '0', '-z', target], logfile)
            run_tool(['ultrahdr_app', '-m', '1', '-j', target, '-P', '-u', '0'], logfile)
            back = work / 'uhdr_back.raw'
            run_tool(['ultrahdr_app', '-m', '1', '-j', target, '-o', '2', '-O', '5',
                      '-u', '0', '-z', back], logfile)
            assert back.stat().st_size == h*w*4
            details['ultrahdr_decoded_bytes'] = back.stat().st_size
            details['files'].append(str(target.relative_to(OUT)))
        else:
            details['errors'].append('Ultra HDR 생략: ultrahdr_app 도구 없음')
    except Exception as exc:
        details['errors'].append('Ultra HDR 실패: ' + str(exc))
    shutil.rmtree(work)
    return details


def time_strip(panoramas, records, coords):
    # A literal centre column of the perspective view: its ordinate is not equal-angle.
    altitude = coords[1][:, coords[1].shape[1]//2]
    cut_alt = altitude[altitude >= 0]
    profiles = np.stack([sample(panoramas[e], np.zeros((len(cut_alt), 1), np.float32),
                                cut_alt[:, None])[:, 0] for e in ES])
    lum = profiles @ Y
    chroma = profiles / np.maximum(lum[..., None], 1e-30)
    width = 1600
    elevations = np.linspace(2, -12, width)
    field = np.empty((len(cut_alt), width, 3), np.float32)
    logkey = np.log([records[str(e)]['central_nit'] for e in ES])
    zero_k = records['0']['fixed_k']
    anchor_key = records['-3']['central_nit']
    for column, e in enumerate(elevations):
        i = min(int(np.searchsorted(-np.array(ES), -e, side='right')-1), len(ES)-2)
        t = (ES[i]-e)/(ES[i]-ES[i+1])
        y = np.exp((1-t)*np.log(lum[i]) + t*np.log(lum[i+1]))
        c = (1-t)*chroma[i] + t*chroma[i+1]
        key = math.exp(float(np.interp(-e, -np.array(ES), logkey)))
        k = zero_k if e >= -3 else zero_k * (anchor_key/key)**ADAPT
        field[:, column] = tone(c*y[:, None], k)
    # Add axes without remapping the actual centre profile to equirectangular coordinates.
    base = Image.fromarray(encode(field))
    plate = Image.new('RGB', (width+90, len(cut_alt)+82), '#0e1220')
    plate.paste(base, (75, 20))
    d = ImageDraw.Draw(plate)
    for e in ES:
        x = 75+round((2-e)/14*(width-1))
        d.line((x, len(cut_alt)+21, x, len(cut_alt)+29), fill='#bbc7dd')
        d.text((min(x-8, width+40), len(cut_alt)+34), f'{e:+d}', fill='#bbc7dd')
    for a in (40, 30, 20, 10, 0):
        row = int(np.argmin(abs(cut_alt-a)))+20
        d.text((8, row-5), f'{a} deg', fill='#bbc7dd')
    d.text((75, len(cut_alt)+59), 'SUN ELEVATION (deg)  /  INTERPOLATED: log luminance + RGB/Y chromaticity', fill='#bbc7dd')
    plate.save(OUT/'full/time_strip.png', compress_level=9)
    # Web copy retains axes; full plate has a 1600px actual colour field.
    plate.resize((1600, round(plate.height*1600/plate.width)), Image.Resampling.LANCZOS).save(
        OUT/'web/time_strip.png', compress_level=9)
    return {'centre_column_altitude_range_deg': [float(cut_alt.min()), float(cut_alt.max())],
            'ordinate': 'perspective centre-column pixels, altitude ticks',
            'interpolation': 'log Y and linear RGB/Y chromaticity', 'exposure': 'b'}


def figure(src, caption, css=''):
    return f'<figure class="{css}"><img src="web/{src}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>'


def build_html(manifest):
    source = manifest['input_directory']
    text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Twilight Sky Renders</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0e1220;color:#e5eaf4;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.7}}
main{{max-width:1100px;margin:auto;padding:40px 16px 72px}}h1{{font-size:clamp(28px,5vw,44px);line-height:1.3;font-weight:650}}h2{{margin:56px 0 20px;font-size:24px;font-weight:600}}h3{{font-size:16px;font-weight:500;color:#cbd4e6}}p{{color:#b7c3d8;max-width:950px}}figure{{margin:0 0 24px;min-width:0}}img{{display:block;max-width:100%;height:auto;border-radius:8px}}figcaption{{color:#abb9cf;font-size:12px;margin-top:8px}}.trio{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.phone{{max-width:300px}}.pair{{display:grid;grid-template-columns:300px minmax(0,1fr);gap:24px;align-items:start}}.note{{font-size:13px}}nav{{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}}a{{color:#b7c8e8}}nav a{{padding:4px 12px;background:#1a2133;border-radius:6px;text-decoration:none}}@media(max-width:720px){{.trio,.pair{{grid-template-columns:1fr}}.phone{{max-width:300px;margin-inline:auto}}main{{padding-top:24px}}}}
</style></head><body><main><h1>박명 하늘, 빛이 물러나는 시간</h1>
<p>자체 구형 successive-orders 해법의 15파장 전 하늘 분광장({source}, normal · AOD 0.15 · 알베도 0.30)을 원근 화면에 옮긴 감상용 <strong>중간본</strong>입니다. 에어로졸 다중 산란은 아직 수렴 전이며, 격자를 두 배로 했을 때 15–39% 변화가 있었습니다. (a)는 장면의 하늘 중앙을 중간 회색으로 맞춘 카메라 자동노출, (b)는 0°의 노출을 −3°까지 유지한 뒤 일부만 순응해 어두워지는 밝기, (c)는 (b)에 −6° 미만에서 명소시·암소시 혼합을 적용한 모습입니다. 출력 백색은 D65이며, 채도 증폭·색조 이동·자동 화이트밸런스는 쓰지 않았습니다.</p>
<nav>'''
    text += ''.join(f'<a href="#e{tag(e)}">{e:+d}°</a>' for e in ES)
    text += '<a href="#time">시간 흐름</a><a href="#compare">기존 LUT 비교</a></nav>'
    for e in ES:
        rec = manifest['scenes'][str(e)]
        zenith = rec['zenith_nit']
        text += f'<section id="e{tag(e)}"><h2>태양고도 {e:+d}°</h2><h3>태양을 향해 · 수평 시야 100°</h3><div class="trio">'
        for mode in MODES:
            caption = f'중간본 · 태양 {e:+d}° · ({mode}) {MODES[mode]} · 천정 {zenith:.5g} cd/m²'
            text += figure(f'{tag(e)}_sun_{mode}.png', caption)
        text += '</div><div class="pair">'
        text += figure(f'{tag(e)}_phone_b.png', f'중간본 · 태양 {e:+d}° · 세로 60° · (b) · 천정 {zenith:.5g} cd/m²', 'phone')
        text += figure(f'{tag(e)}_anti_b.png', f'중간본 · 태양 {e:+d}° · 태양 반대쪽 100° · (b) · 천정 {zenith:.5g} cd/m²')
        text += '</div></section>'
    text += '<section id="time"><h2>시간 흐름</h2><p class="note">태양 쪽 가로 화면의 가운데 세로줄을 +2°부터 −12°까지 펼쳤습니다. 여섯 고도 사이의 색은 로그 휘도와 색도 보간입니다. 세로축은 원근 화면의 시선고도, 밝기는 (b)이며 시간 간격은 태양고도 기준입니다.</p>'
    text += figure('time_strip.png', '중간본 · 태양 +2° → −12° · (b) · 보간 · 천정 340.18 → 0.0021638 cd/m²')
    text += '</section><section id="compare"><h2>기존 앱 LUT와 같은 화면, 같은 노출</h2><p class="note">위는 새 SOS 중간본, 아래는 기존 앱 twilight_lut_v1입니다. 각 쌍은 같은 원근 투영과 SOS에서 정한 (b) 노출·톤 곡선을 공유합니다. 기존 LUT를 따로 밝게 맞추지 않았습니다.</p>'
    for e in ES:
        r = manifest['scenes'][str(e)]
        text += figure(f'{tag(e)}_compare.png', f'태양 {e:+d}° · 위 SOS 중간본 / 아래 앱 LUT · (b) · 천정 SOS {r["zenith_nit"]:.5g} / LUT {r["app_zenith_nit"]:.5g} cd/m²')
    text += '</section><p class="note">감상용 노출은 실제 모니터에서 눈이 순응하는 과정의 측정값이 아닙니다. 푸르키네 판은 기존 혼합식을 D65 무채색으로 표시한 근사입니다. 원본 크기 PNG와 HDR 파일은 이 폴더의 full / hdr에 별도로 보관했습니다.</p></main></body></html>'
    (OUT/'index.html').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-hdr', action='store_true', help='SDR preview pass only; final pass must omit this flag')
    args = parser.parse_args()
    started = time.monotonic()
    for folder in ('full', 'web', 'hdr'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    # Decide once from currently completed real artifacts; do not wait for stage 3.
    newer = ROOT/'results/sos_sky2'
    complete = all((newer/f'{kind}_e{e}.npy').is_file() for e in ES for kind in ('spec', 'rgb')) and (newer/'rgb_weights.npy').is_file()
    source = newer if complete else ROOT/'results/sos_sky'
    log('입력 ' + str(source.relative_to(ROOT)))
    weights = np.load(source/'rgb_weights.npy')
    cie = json.loads((ROOT/'assets/lef_cie.json').read_text())
    spectra, panoramas, purkinjes = {}, {}, {}
    manifest = {'risk_class': 'D0', 'purpose': '감상용 중간본',
                'input_directory': str(source.relative_to(ROOT)),
                'stage3_panorama_complete_at_selection': complete, 'input_hashes': {},
                'scenes': {}, 'hdr': {}, 'visual_inspection': 'pending',
                'exposure': {'adapt_log_slope': ADAPT, 'reference_e': 0, 'adapt_anchor_e': -3,
                             'central_region': 'x=.40..60W, y=.25..55H of sun perspective sky',
                             'curve': 'Hable .15,.50,.10,.20,.02,.30; .98 asymptote; scalar max-channel shoulder',
                             'fixed_formula': 'k=k0 (e>=-3); k=k0*(L(-3)/L(e))^.70 (e<-3)'},
                'projections': VIEWS, 'threads': 'numpy/Pillow 1; avifenc/avifdec 4; libultrahdr 2.0.2 capped at 4; sequential',
                'omissions': []}
    paths = [source/'rgb_weights.npy'] + [source/f'{kind}_e{e}.npy' for e in ES for kind in ('spec', 'rgb')]
    for p in paths:
        manifest['input_hashes'][str(p.relative_to(ROOT))] = digest(p)
    for e in ES:
        spec, rgb = np.load(source/f'spec_e{e}.npy'), np.load(source/f'rgb_e{e}.npy')
        assert spec.shape == (100, 360, 15) and rgb.shape == (100, 360, 3)
        assert np.isfinite(spec).all() and np.isfinite(rgb).all() and (spec >= 0).all()
        reconstructed = spec @ weights
        relative = float(np.linalg.norm(reconstructed-rgb)/np.linalg.norm(rgb))
        assert relative < 1e-5
        assert (rgb >= 0).all(), 'Negative input RGB requires explicit gamut policy'
        spectra[e], panoramas[e] = spec, rgb*NIT
        purkinjes[e], info = purkinje(panoramas[e], spec, cie, e)
        manifest['scenes'][str(e)] = {'zenith_nit': float((panoramas[e][0]@Y).mean()),
                                      'rgb_reconstruction_relative_error': relative, 'purkinje': info}
    coords = {view: projection(*values) for view, values in VIEWS.items()}
    for e in ES:
        reference = project(panoramas[e], coords['sun'])
        key, k = central(reference)
        manifest['scenes'][str(e)].update(central_nit=key, auto_k=k)
    k0 = manifest['scenes']['0']['auto_k']
    anchor = manifest['scenes']['-3']['central_nit']
    manifest['exposure'].update(k0=k0, anchor_nit=anchor)
    log(f'고정 기준 k0={k0:.8g}; L(-3)={anchor:.6g} cd/m²')
    lut = APP['app_lut']()
    lut_path = machine_path('app_lut')
    manifest['app_lut'] = {'path': str(lut_path), 'sha256': digest(lut_path), 'axes': [lut[1], lut[2].tolist(), lut[3].tolist()]}
    for e in ES:
        record = manifest['scenes'][str(e)]
        kb = k0 if e >= -3 else k0*(anchor/record['central_nit'])**ADAPT
        record['fixed_k'] = kb
        record['views'] = {}
        for view in VIEWS:
            log(f'태양 {e:+d}° · {view} 렌더')
            rgb = project(panoramas[e], coords[view])
            mes = project(purkinjes[e], coords[view]) if e < -6 else rgb
            key, ka = central(rgb)
            record['views'][view] = {'central_nit': key, 'auto_k': ka}
            for mode in MODES:
                plate = tone(mes if mode == 'c' else rgb, ka if mode == 'a' else kb)
                name = f'{tag(e)}_{view}_{mode}.png'
                save_png(OUT/'full'/name, plate)
                if view == 'sun' or mode == 'b':
                    width = 960 if view != 'phone' else 480
                    save_png(OUT/'web'/name, resize_linear(plate, width))
                if view == 'sun' and mode == 'b':
                    record['display_b_median_code'] = float(np.median(encode(plate)[..., :][coords[view][1]>=0]))
                    sos_b = plate
            if view == 'sun':
                app_pano = APP['app_pano'](e, lut).astype(np.float32)*NIT
                app = tone(project(app_pano, coords['sun']), kb)
                record['app_zenith_nit'] = float((app_pano[0]@Y).mean())
                comparison = np.concatenate((sos_b, app), axis=0)
                save_png(OUT/'full'/f'{tag(e)}_compare.png', comparison)
                save_png(OUT/'web'/f'{tag(e)}_compare.png', resize_linear(comparison, 960))
                if not args.no_hdr:
                    manifest['hdr'][str(e)] = export_hdr(rgb, kb, f'{tag(e)}_sun_b')
        log(f'태양 {e:+d}° 완료 · k(b)={kb:.6g}')
    manifest['time_strip'] = time_strip(panoramas, manifest['scenes'], coords['sun'])
    web_files = list((OUT/'web').glob('*.png'))
    manifest['web_png_bytes'] = sum(p.stat().st_size for p in web_files)
    assert len(web_files) == 37
    assert manifest['web_png_bytes'] <= 14_000_000, manifest['web_png_bytes']
    if args.no_hdr:
        manifest['omissions'].append('임시 SDR 확인 실행: HDR는 최종 실행에서 생성')
    for e, item in manifest['hdr'].items():
        manifest['omissions'].extend(f'e={e}: {err}' for err in item['errors'])
    for p in paths:
        assert digest(p) == manifest['input_hashes'][str(p.relative_to(ROOT))], f'Input changed: {p}'
    assert digest(lut_path) == manifest['app_lut']['sha256']
    build_html(manifest)
    manifest['elapsed_seconds'] = time.monotonic()-started
    manifest['file_inventory'] = [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size}
                                  for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    log(f'완료 {manifest["elapsed_seconds"]:.1f}s · 웹 PNG {manifest["web_png_bytes"]:,} bytes')


if __name__ == '__main__':
    main()
