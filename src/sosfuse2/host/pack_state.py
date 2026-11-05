"""Portable preview + atomic archive; only the four final products are packed."""
import ast, json, math, os, resource, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np
from PIL import Image, PngImagePlugin
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREVIEW_PNG_LEVEL=6

def render_functions():
    path=ROOT/'accepted/sos_render.py';tree=ast.parse(path.read_text())   # frozen copy of sky-sos scripts/sos_render.py
    names={'film','tone','encode','save_png','projection','cubic_weights','sample','project','central'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise RuntimeError('E_RENDER_ADAPTER_DRIFT')
    scope=dict(np=np,math=math,Image=Image,PngImagePlugin=PngImagePlugin,Y=np.array([.2126,.7152,.0722]))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    encode=scope['encode']
    def save_png(path,rgb,seed=17):
        # sos_render.save_png with PNG level 6 instead of 9: identical pixels, 0.98 s -> 0.03 s per preview, +8 % bytes
        # (2026-10-25, sky-sos results/finalize_20261025). Preview bytes are not a contract; table.npz stays as it was.
        path.parent.mkdir(parents=True,exist_ok=True);info=PngImagePlugin.PngInfo();info.add(b'sRGB',b'\x00')
        Image.fromarray(encode(rgb,seed)).save(path,compress_level=PREVIEW_PNG_LEVEL,pnginfo=info)
    scope['save_png']=save_png;return scope

def previews(out):
    funcs=render_functions();weights=np.load(ROOT/'assets/rgb_weights.npy')
    with np.load(out/'table.npz') as f:table=f['radiance'];wl=f['wavelength_nm'];hs=f['solar_elevation_deg']
    if len(wl)==15:rgb=table@weights
    elif len(wl)==1:rgb=np.repeat(table,3,axis=-1) # validation preview is explicitly monochrome
    else:raise RuntimeError('E_PREVIEW_SPECTRUM: unsupported bin count')
    for h in (-3,-6):
        hit=np.flatnonzero(hs==h)
        if len(hit)!=1:raise RuntimeError(f'E_PREVIEW_ALTITUDE: missing {h}')
        folded=rgb[hit[0]]
        # Renderer sample expects azimuth -179.5..179.5 (sun at central columns).
        pano=np.concatenate([folded[:,::-1],folded],axis=1)
        projected=funcs['project'](pano,funcs['projection'](960,540,100,0))
        if (projected@funcs['Y']).max()>0:_,k=funcs['central'](projected)
        else:k=1.
        funcs['save_png'](out/f'preview_m{abs(h):02d}.png',funcs['tone'](projected,k))

def pack_state(out,pack,compression='auto',started=None,*,reuse_previews=False):
    """Reuse previews only after a successful pack in the same attempt."""
    meta=json.loads((out/'meta.json').read_text())
    if not meta.get('complete'):raise RuntimeError('E_PACK_INCOMPLETE')
    if not reuse_previews:previews(out)
    pack.mkdir(parents=True,exist_ok=True)
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    meta['peak_rss_bytes']=max(meta.get('peak_rss_bytes',0),peak)
    if started is not None:
        meta['elapsed_seconds']=time.monotonic()-started
        meta['elapsed_scope']='current successful attempt through previews; archive compression excluded'
    meta_tmp=(out/'meta.json').with_suffix('.json.tmp')
    meta_tmp.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n');os.replace(meta_tmp,out/'meta.json')
    zstd=shutil.which('zstd')
    if compression=='zstd' and not zstd:raise RuntimeError('E_ZSTD_MISSING: install zstd (required by pack and verify)')
    use_zstd=bool(zstd) and compression!='gzip';suffix='.tar.zst' if use_zstd else '.tar.gz'
    target=pack/(out.name+suffix);tmp=target.with_suffix(target.suffix+'.tmp')
    names=('table.npz','meta.json','preview_m03.png','preview_m06.png')
    try:
        if use_zstd:
            with tmp.open('wb') as f:
                process=subprocess.Popen([zstd,'-q','-T'+os.environ.get('OMP_NUM_THREADS',str(os.cpu_count() or 1)),'-c'],stdin=subprocess.PIPE,stdout=f)
                try:
                    with tarfile.open(fileobj=process.stdin,mode='w|') as tar:
                        for name in names:tar.add(out/name,arcname=name,recursive=False)
                finally:process.stdin.close()
                if process.wait()!=0:raise RuntimeError('E_PACK_ZSTD_FAILED')
        else:
            with tarfile.open(tmp,'w:gz') as tar:
                for name in names:tar.add(out/name,arcname=name,recursive=False)
        os.replace(tmp,target)
    finally:tmp.unlink(missing_ok=True)
    print(f'PACK {target.name} {target.stat().st_size} bytes',flush=True);return target
