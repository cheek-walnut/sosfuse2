"""One final numerical validation checkpoint; accepted reference is read-only.

E6 (selftest, full_35, spectral_15) and G-H pass against the record RECORD (HEAD NumPy, user decision DECISION, 2026-10-22;
results/bless_20261022/SPEC_E6GH.md). The old references stay computed as information only (historical_delta, never in pass_):
the old CPU-engine fixture LEGACY (also rig/run_state.py and bootstrap_v3.sh:85, unchanged) and the original sos_full CPU
transport (original_kernel). Task B's Simpson prefix moved fused results away from both by design (bless REPORT section 4).
"""
import hashlib
import io
import json
import time
import numpy as np
from .fused import Solver, load_engine
from .geometry import create, TransientGeometry, normal_state
from . import arithmetic as a
from . import paths
from . import reference
from .timers import TIMERS

# The old CPU-engine fixture lives outside git in the sky-sos checkout (rig/selftest_ref.npz); the package reaches it only
# through SOSFUSE2_SKY_SOS_ROOT. Without it the record's own selftest query directions are used and the informational
# historical_delta is omitted (public split).
_SKY=paths.sky_sos_root()
LEGACY=(_SKY/'rig/selftest_ref.npz') if _SKY is not None else None
LEGACY_SHA256='a48688cdc86bf6cb88950944840b88bc846a813864a54ad4672c9ee034ef03be'
LEGACY_META=(_SKY/'rig/selftest_ref_meta.json') if _SKY is not None else paths.FIXTURES/'selftest_ref_meta.json'
RECORD=paths.FIXTURES/'selftest_ref_simpson.npz'
RECORD_META=paths.FIXTURES/'selftest_ref_simpson_meta.json'


def legacy_available():
    return LEGACY is not None and LEGACY.is_file()
# Pinned once (results/e6gh_20261022/make_record.py, cq mac-cpu); a new record needs a new user decision ID.
RECORD_SHA256='2543ab1d296780e408cd93f8091bffc118cacbace6d11b3d47306df5471c58b4'
DECISION='taskB1-simpson-tol001-20261022'
TOLERANCE=1e-5
SPECTRAL_WAVELENGTHS=360+(np.arange(15)+.5)*470/15


def relative(actual, expected):
    return float(np.max(abs(actual-expected)/np.maximum(abs(expected),1e-30)))


def versus(actual, expected):
    """relative() against a record array; a missing array or another shape gives None (never passes)."""
    if expected is None or np.shape(actual)!=np.shape(expected):return None
    return relative(actual,expected)


def within(error):
    return bool(error is not None and np.isfinite(error) and error<=TOLERANCE)


def worst(actual, expected):
    """Diagnostic for an informational delta: where the largest relative difference sits and how small its denominator is."""
    actual=np.asarray(actual,'f8');expected=np.asarray(expected,'f8');rel=abs(actual-expected)/np.maximum(abs(expected),1e-30)
    i=np.unravel_index(int(np.argmax(rel)),rel.shape);largest=float(np.max(abs(expected)))
    return dict(index=[int(x) for x in i],relative=float(rel[i]),actual=float(actual[i]),expected=float(expected[i]),
                expected_max_abs=largest,expected_fraction_of_max=float(abs(expected[i])/largest) if largest>0 else 0.)


def load_record():
    """The E6/G-H pass reference. A missing file stops (E_SELFTEST_RECORD_MISSING); a wrong sha fails the identity."""
    if not RECORD.is_file():raise RuntimeError(f'E_SELFTEST_RECORD_MISSING: {RECORD} (decision {DECISION})')
    data=RECORD.read_bytes()
    with np.load(io.BytesIO(data),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    contract=json.loads(str(arrays.pop('contract'))) if 'contract' in arrays else {}
    digest=hashlib.sha256(data).hexdigest()
    return dict(arrays=arrays,contract=contract,sha256=digest,sha_ok=digest==RECORD_SHA256)


def record_identity(rec, item, settings, inputs):
    """Gating input identity: pinned record sha, decision, today's normal state, this item's settings and query inputs."""
    c=rec['contract'];bad=[]
    if not rec['sha_ok']:bad.append('sha256')
    if c.get('decision')!=DECISION:bad.append('decision')
    if c.get('state')!=json.loads(json.dumps(normal_state())):bad.append('state')
    if (c.get('settings') or {}).get(item)!=json.loads(json.dumps(settings)):bad.append('settings')
    for k,v in inputs.items():
        if not np.array_equal(rec['arrays'].get(f'{item}_{k}'),np.asarray(v)):bad.append(k)
    return dict(pass_=not bad,mismatch=bad,file=paths.rel(RECORD),sha256=rec['sha256'],pinned_sha256=RECORD_SHA256,
                decision=c.get('decision'),contract='pinned record sha + decision + normal state + settings + query inputs')


def moment_comparison(actual, expected):
    """Compare coefficient vectors (axis 0), independently per wavelength.

    Float32 accumulation and cancellation do not support a uniform 1e-6
    componentwise relative gate near zero, even with Kahan summation. Floor
    each denominator at 1e-3 of its own vector's maximum magnitude; require
    scaled error <= 1e-5 and L2 relative error <= 1e-6. Physical accuracy is
    checked separately by the reference selftest and the 35-direction test.
    """
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    delta=abs(actual-expected);magnitude=abs(expected)
    vector_max=np.max(magnitude,axis=0,keepdims=True)
    scale=np.maximum(magnitude,1e-3*vector_max)
    # Exact zero vectors match only if the corresponding actual vector is zero.
    scaled=np.divide(delta,scale,out=np.where(delta==0,0.,np.inf),where=scale>0)
    expected_l2=np.linalg.norm(expected,axis=0);delta_l2=np.linalg.norm(delta,axis=0)
    l2=np.divide(delta_l2,expected_l2,out=np.where(delta_l2==0,0.,np.inf),where=expected_l2>0)
    raw=delta/np.maximum(magnitude,1e-30)
    index=np.unravel_index(np.argmax(raw),raw.shape)
    maximum=float(np.max(scaled));l2_maximum=float(np.max(l2))
    largest=float(vector_max[(0,)+index[1:]])
    return dict(moment_max_relative=float(np.max(raw)),moment_scaled_relative=maximum,
                moment_l2_relative=l2_maximum,moment_scale_epsilon=1e-3,
                moment_scaled_tolerance=1e-5,moment_l2_tolerance=1e-6,
                moment_worst_relative=dict(index=[int(i) for i in index],
                                           numpy_abs=float(magnitude[index]),vector_max_abs=largest,
                                           numpy_fraction_of_vector_max=float(magnitude[index]/largest) if largest>0 else 0.),
                moment_pass_=bool(np.isfinite(maximum) and maximum<=1e-5 and
                                  np.isfinite(l2_maximum) and l2_maximum<=1e-6))


def original_kernel(output):
    """The accepted CPU transport kernel (native/sh_cpu.cpp, built into the work root by accepted.engine.compile_kernel)."""
    return load_engine().Kernel(4,1.2,.8)


def cases():
    result={}
    # Coordinate cancellation at Earth radius and nearly tangent paths.
    x=np.array([6371.+.000001,6371.+.1,6371.+80.])
    height=a.rounded(a.sub(a.split(x),(a.R,a.F(0))))
    error=float(np.max(abs(height-(x-6371.))))
    p=np.array([[6371.1,0.,0.]]);d=np.array([[-.005,np.sqrt(1-.005**2),0.]])
    t=-np.sum(p*d,axis=-1);q=a.add(a.split(p),a.mul(a.split(t[:,None]),a.split(d)))
    q2=a.mul(q,q);rr=a.root(a.add(a.add((q2[0][:,0],q2[1][:,0]),(q2[0][:,1],q2[1][:,1])),(q2[0][:,2],q2[1][:,2])))
    h=a.rounded(a.sub(rr,(a.R,a.F(0))));expected=np.linalg.norm(p+t[:,None]*d,axis=-1)-6371.
    error=max(error,float(np.max(abs(h-expected))))
    result['coordinate_cancellation']=dict(max_absolute=error,pass_=error<2e-7)
    # A nonpositive weighted corner selects the linear fallback; a zero-weight corner is ignored (gate v2 2026-10-21,
    # reference.transport / kernel.cu bad_trace). trace_scale itself only applies the resulting per-sample flags.
    value=np.array([[[2.,3.,5.,.7]], [[-1.,2.,0.,-.4]], [[2.,3.,5.,.7]]],'f4')
    actual=reference.trace_scale(value,np.log(np.array([[20.],[9.],[20.]],'f4')),np.array([[True],[False],[False]]))
    expected=np.array([[[4.,6.,10.,1.4]], [[-1.,2.,0.,-.4]], [[2.,3.,5.,.7]]],'f4')
    error=float(np.max(abs(actual-expected)))
    result['fallback_rule']=dict(max_absolute=error,pass_=error<2e-6,
                                  rule='positive traces use geometric mean; a nonpositive weighted corner uses linear fallback; zero-weight corners are ignored')
    y=a.basis(np.array([0.,1.,-1.]));expected=np.array([1/np.sqrt(4*np.pi),0.,-np.sqrt(3/(4*np.pi))])
    error=float(np.max(abs(y[0,:3]-expected)))
    result['SH_low_orders']=dict(max_absolute=error,pass_=error<1e-7)
    # The actual 32-lane shuffle tree, checked with independent explicit nodes.
    x=np.arange(33,dtype='f4')[None,:,None];got=a.warp_sum(x)[0,0]
    result['warp_tree']=dict(sum=float(got),pass_=bool(got==528))
    # Each adjacent stencil retains every old source cell it still references.
    # Symbolic coefficient values establish the contract without recomputing SH.
    moves=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(1,1)]
    nc=20;old=np.tile(np.array([47,48,67,68],'i4'),(len(moves),1))
    cache=old[:,:,None,None].astype('f4')*a.F(1000)+np.arange(33,dtype='f4')[None,None,:,None]
    previous=cache.copy();tags=old.copy();base=np.array([47+dr*nc+dc for dr,dc in moves],'i4')
    cells=base[:,None]+np.array([0,1,nc,nc+1],'i4')[None,:];valid=np.ones(len(moves),bool);valid[-1]=False
    reference.reuse_corners(cache,tags,dict(cells=cells,valid=valid),nc)
    okay=np.array_equal(cache[-1],previous[-1]) and np.array_equal(tags[-1],old[-1])
    for ray in range(len(moves)-1):
        for corner,wanted in enumerate(cells[ray]):
            matches=np.flatnonzero(old[ray]==wanted)
            if len(matches):okay &= tags[ray,corner]==wanted and np.array_equal(cache[ray,corner],previous[ray,matches[0]])
    result['adjacent_corner_aliases']=dict(cases=9,pass_=bool(okay))
    return dict(pass_=all(r['pass_'] for r in result.values()),cases=result)


def legacy_selftest():
    """Query directions of the reduced selftest and the historical values: the old CPU-engine fixture when reachable,
    otherwise the pinned record's own selftest arrays (same directions; identity 'record')."""
    if legacy_available():
        with np.load(LEGACY,allow_pickle=False) as z:
            return dict(radiance=z['radiance'],el=z['view_elevation_deg'],az=z['azimuth_deg'],hs=float(z['solar_elevation_deg']),identity=str(z['identity']))
    with np.load(RECORD,allow_pickle=False) as z:
        return dict(radiance=z['selftest_radiance'],el=z['selftest_view_elevation_deg'],az=z['selftest_azimuth_deg'],
                    hs=float(z['selftest_solar_elevation_deg']),identity='record')


def reduced_values(output,backend,block_size):
    """Reduced selftest values (two applies, legacy query directions); no comparison."""
    tt=time.perf_counter();s,scope,settings=create(output/'selftest_geometry','selftest',[550.],backend,block_size)
    geometry=TransientGeometry(s,scope);solver=Solver(s,geometry)
    field=s.kernel.xp.zeros((s.geo.ns,571,1),'f4')
    for _ in range(2):field=solver.apply(field)
    packed=s.kernel.host(field);full=np.zeros((s.geo.ns,571,2),'f4');full[...,0]=packed[...,0]
    L=legacy_selftest();el,az,hs=L['el'],L['az'],L['hs']
    value=(s.query(hs,el,az,full,True)+s.query(hs,el,az,full,False))[...,0]
    return dict(value=value,packed=packed,settings=settings,legacy=L,seconds=time.perf_counter()-tt,
                apply_seconds=solver.timings['apply_seconds'],geometry=geometry.stats)


def legacy_identity(settings):
    """The old fixture's identity bound to today's state and settings (now informational: it attests the historical delta)."""
    meta=json.loads(LEGACY_META.read_text())
    if not legacy_available(): return None,meta        # informational delta unavailable outside sky-sos
    # The fixture records historical code provenance, not a requirement that
    # today's scheduler/bootstrap have identical bytes. Pin the measured data
    # and bind its original identity to today's state and numerical settings.
    identity=hashlib.sha256(json.dumps(dict(state=normal_state(),settings=settings,hashes=meta['hashes']),sort_keys=True).encode()).hexdigest()
    fixture_ok=hashlib.sha256(LEGACY.read_bytes()).hexdigest()==LEGACY_SHA256
    with np.load(LEGACY,allow_pickle=False) as z:ident=str(z['identity'])
    return fixture_ok and ident==identity and settings==meta['settings'] and normal_state()['parameters']==meta['parameters'],meta


def reduced(output,backend,block_size):
    tt=time.perf_counter();m=reduced_values(output,backend,block_size);value,L=m['value'],m['legacy']
    rec=load_record();expected=rec['arrays'].get('selftest_radiance')
    ident=record_identity(rec,'selftest',m['settings'],dict(view_elevation_deg=L['el'],azimuth_deg=L['az'],solar_elevation_deg=L['hs']))
    error=versus(value,expected);legacy_ok,meta=legacy_identity(m['settings'])
    np.savez(output/'selftest_values.npz',radiance=value,reference=L['radiance'],record=value*np.nan if expected is None else expected,source=m['packed'])
    report=dict(max_relative=error,tolerance=TOLERANCE,pass_=within(error) and ident['pass_'],
                identity_match=ident['pass_'],identity_contract=ident['contract'],record=ident,
                historical_delta=None if legacy_ok is None else
                                 dict(blocking=False,reference=f'{LEGACY} (old CPU engine, sha {LEGACY_SHA256[:12]})',
                                      max_relative=relative(value,L['radiance']),worst=worst(value,L['radiance']),
                                      identity_match=legacy_ok,reference_measured_utc=meta['measured_utc']),
                seconds=time.perf_counter()-tt,settings=m['settings'],apply_seconds=m['apply_seconds'],geometry=m['geometry'])
    return report,m['packed']


# The accepted converged full source (sky-sos results/, 2026-09) is research material, reachable only via SOSFUSE2_SKY_SOS_ROOT.
FULL_SOURCE=(_SKY/'results/sos_full/b2_run/base550_d0.03_1_684f84f1be89fb8c/source.npz') if _SKY is not None else None


def full_queries_values(output,backend,block_size):
    """35 queries on the accepted converged source: fused values and the original CPU transport (historical); no comparison."""
    if FULL_SOURCE is None:raise RuntimeError('E_SKY_SOS_ROOT: full_35 needs SOSFUSE2_SKY_SOS_ROOT')
    tt=time.perf_counter();s,scope,settings=create(output/'full_query_geometry','full',[550.],backend,block_size)
    artifact=FULL_SOURCE
    with np.load(artifact,allow_pickle=False) as z:
        field=z['source']
        if not np.array_equal(z['h'],s.geo.h) or not np.array_equal(z['chi'],s.geo.chi):raise RuntimeError('E_REFERENCE_GRID: full source grid differs')
    cpu=original_kernel(output);kernel=s.kernel;fuse_pack=s.geo.pack
    el=np.array([2.,10.,30.,10.,30.,10.,89.]);az=np.array([0.,0.,0.,180.,180.,90.,0.]);hs=(2,-3,-6,-9,-12)
    actual={};expected={}
    for first,name in ((True,'single'),(False,'multiple')):
        s.kernel=kernel;s.geo.pack=fuse_pack
        actual[name]=np.stack([s.query(h,el,az,field,first)[...,0] for h in hs])
        s.kernel=cpu;s.geo.pack=lambda p,d,n:scope['pack'](p,d,s.geo.h,s.geo.chi,s.geo.sun,n)
        expected[name]=np.stack([s.query(h,el,az,field,first)[...,0] for h in hs])
    s.kernel=kernel;s.geo.pack=fuse_pack
    return dict(actual=actual,cpu=expected,settings=settings,el=el,az=az,hs=np.array(hs,float),
                source_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),seconds=time.perf_counter()-tt)


def full_queries(output,backend,block_size):
    tt=time.perf_counter();m=full_queries_values(output,backend,block_size);actual,cpu=m['actual'],m['cpu']
    rec=load_record()
    ident=record_identity(rec,'full_35',m['settings'],dict(solar_elevation_deg=m['hs'],view_elevation_deg=m['el'],azimuth_deg=m['az'],
                                                           source_sha256=m['source_sha256']))
    record={name:rec['arrays'].get(f'full_35_{name}') for name in actual}
    errors={name:versus(actual[name],record[name]) for name in actual}
    np.savez(output/'full_35_values.npz',**actual,reference_single=cpu['single'],reference_multiple=cpu['multiple'],
             **{f'record_{n}':actual[n]*np.nan if v is None else v for n,v in record.items()},
             solar_elevation_deg=m['hs'],view_elevation_deg=m['el'],azimuth_deg=m['az'])
    return dict(max_relative=errors,tolerance=TOLERANCE,pass_=ident['pass_'] and all(within(v) for v in errors.values()),record=ident,
                historical_delta=dict(blocking=False,reference='original sos_full CPU transport via rig adapter (original_kernel)',
                                      max_relative={n:relative(actual[n],cpu[n]) for n in actual},worst={n:worst(actual[n],cpu[n]) for n in actual}),
                seconds=time.perf_counter()-tt,settings=m['settings'],source_sha256=m['source_sha256'],
                scope='550 nm, 7 directions x 5 solar elevations; accepted converged source shared; pass vs the record, original transport historical')


def spectral_values(output,field,block_size):
    """The 15-lane log operator at an actual varying spectral optical depth: fused values and the original CPU transport."""
    wavelengths=SPECTRAL_WAVELENGTHS
    s,scope,settings=create(output/'spectral_geometry','selftest',wavelengths,'numpy',block_size)
    full=np.zeros((s.geo.ns,571,30),'f4');full[...,::2]=np.tile(field,(1,1,15))
    fused=s.kernel;pack_fused=s.geo.pack;cpu=original_kernel(output)
    el=np.array([2.,30.,89.]);az=np.array([0.,90.,180.]);actual={};expected={}
    for first in (True,False):
        s.kernel=fused;s.geo.pack=pack_fused;actual[first]=s.query(-6,el,az,full,first)[...,0]
        s.kernel=cpu;s.geo.pack=lambda p,d,n:scope['pack'](p,d,s.geo.h,s.geo.chi,s.geo.sun,n)
        expected[first]=s.query(-6,el,az,full,first)[...,0]
    return dict(actual=actual,cpu=expected,settings=settings,wavelength_nm=wavelengths,el=el,az=az,hs=-6.)


def spectral(output,field,block_size):
    """The 15-lane log operator at an actual varying spectral optical depth."""
    m=spectral_values(output,field,block_size);rec=load_record();reports=[]
    ident=record_identity(rec,'spectral_15',m['settings'],dict(wavelength_nm=m['wavelength_nm'],view_elevation_deg=m['el'],
                                                               azimuth_deg=m['az'],solar_elevation_deg=m['hs']))
    names={True:'first',False:'multiple'};saved={}
    for first in (True,False):
        actual,cpu=m['actual'][first],m['cpu'][first];expected=rec['arrays'].get(f'spectral_15_{names[first]}');error=versus(actual,expected)
        saved.update({names[first]:actual,f'reference_{names[first]}':cpu,f'record_{names[first]}':actual*np.nan if expected is None else expected})
        reports.append(dict(first=first,wavelengths=15,max_relative=error,tolerance=TOLERANCE,pass_=within(error),
                            historical_delta=dict(blocking=False,reference='original sos_full CPU transport via rig adapter (original_kernel)',
                                                  max_relative=relative(actual,cpu),worst=worst(actual,cpu))))
    np.savez(output/'spectral_15_values.npz',**saved,wavelength_nm=m['wavelength_nm'],view_elevation_deg=m['el'],azimuth_deg=m['az'],
             solar_elevation_deg=m['hs'],index_order='(direction, wavelength); direction i = (view_elevation_deg[i], azimuth_deg[i])')
    return dict(pass_=ident['pass_'] and all(bool(r['pass_']) for r in reports),record=ident,cases=reports)


def parity(output,block_size,field):
    """Rig-only: actual CUDA vs same-operation NumPy, STRATO off/on, both 1 and 15 lanes."""
    reports=[]
    for enabled,W in ((False,1),(False,15),(True,1),(True,15)):
        wavelengths=[550.] if W==1 else (360+(np.arange(15)+.5)*470/15).tolist()
        import copy
        state=copy.deepcopy(normal_state())
        if enabled:state['stratosphere'].update(status='enabled');state['parameters']['stratosphere_tau550']=.003
        s,scope,_=create(output/f'parity_{W}_strato{int(enabled)}','selftest',wavelengths,'cupy',block_size,state=state)
        geometry=TransientGeometry(s,scope);g=geometry.cell(12);c=s.kernel.setup(s,g['nq'])
        from .stratosphere import source_size
        src=np.zeros((s.geo.ns,source_size(s),W),'f4');src[:,:571]=np.tile(field,(1,1,W))
        if enabled:src[:,571:]=src[:,10:571]*np.float32(.1)
        from .geometry import Bundle
        bundle=Bundle([g],s.kernel.xp)
        for first in (True,False):
            ref,Y=reference.transport(g,c,src,first,return_basis=True)
            moment,flux=reference.project(g,ref,c,Y)
            gm,gf,gq=s.kernel.launch(bundle,s,s.kernel.xp.asarray(src),first,phase=2)
            qerror=relative(s.kernel.host(gq),ref)
            gm,gf,_=s.kernel.launch(bundle,s,s.kernel.xp.asarray(src),first,phase=0)
            mactual=s.kernel.host(gm)[0]
            comparison=moment_comparison(mactual,moment)
            ferr=relative(s.kernel.host(gf)[0],flux)
            reports.append(dict(wavelengths=W,first=first,stratosphere_enabled=enabled,computed_tau550=.003 if enabled else 0.,ray_max_relative=qerror,
                                **comparison,ground_max_relative=ferr,tolerance=1e-6,
                                pass_=qerror<=1e-6 and comparison['moment_pass_'] and ferr<=1e-6))
    return dict(pass_=all(r['pass_'] for r in reports),cases=reports)


def validate(output,backend='numpy',full=True,block_size=128):
    from .provenance import capture,same
    load_record()                                   # E_SELFTEST_RECORD_MISSING before any work
    before=capture()
    report=dict(backend=backend,risk_class='D1',arithmetic_cases=cases())
    TIMERS.reset();timings={}
    report['selftest'],field=reduced(output,backend,block_size);timings['selftest']=TIMERS.snapshot()
    if full:
        report['full_35']=full_queries(output,backend,block_size);timings['full_35']=TIMERS.snapshot()
        report['spectral_15']=spectral(output,field,block_size);timings['spectral_15']=TIMERS.snapshot()
    # L3 (gate v2, results/gate_redesign_20261021/SPEC_V2.md 8): CUDA/NumPy parity is a diagnostic record, not part of
    # pass; sosfuse2.gate G-P/G-T block CUDA vs NumPy instead (atan2f ULP, accumulation order).
    if backend=='cupy':report['cuda_numpy_parity']=dict(parity(output,block_size,field),blocking=False);timings['parity']=TIMERS.snapshot()
    # A1: stage timings are a separate file; the validation gate is unchanged.
    (output/'timings.json').write_text(json.dumps(timings,indent=2,sort_keys=True)+'\n')
    after=capture();report['reference_inputs_unchanged']=same(before,after)
    (output/'reference_inputs_sha256.json').write_text(json.dumps(before,indent=2)+'\n')
    report['pass']=report['reference_inputs_unchanged']['pass_'] and report['arithmetic_cases']['pass_'] and report['selftest']['pass_'] and (not full or report['full_35']['pass_'] and report['spectral_15']['pass_'])
    return report
