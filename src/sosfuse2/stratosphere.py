"""Independent scalar-Mie optical layer. No persistent geometry changes.

Public build_optics(state, h_km, wavelengths_nm) -> None at tau=0, otherwise
an optics dict with beta_e/beta_s [nr,W], eigen [33,W], chi [33,W],
ext/sca [W] (per unit normalized profile), density [nr] and meta.
attach(s,scope) refreshes accepted first-ground illumination once per solver.
All distances are km; beta is km^-1; AOD is dimensionless.
"""
import hashlib,json,math,os
from pathlib import Path
import numpy as np
DEFAULT=Path(__file__).with_name('assets')/'strato_sulfate_rg0.08_sg1.6_n1.43.json'
# State parameters the attach geometry never reads (_geometry_key, run_gpu_state_v3.attach_lock_name): optics
# combined after attach; ssa/g/angstrom/ozone_du were byte-equal on the full grid.
ATTACH_OPTICS_PARAMS=('aod550','stratosphere_tau550','albedo','ssa','g','angstrom','ozone_du','no2_du','o4')


def tau_of(state):
    layer=state.get('stratosphere',{})
    if not isinstance(layer,dict) or layer.get('status')!='enabled':return 0.
    try:tau=float(state.get('parameters',state).get('stratosphere_tau550',0.))
    except (TypeError,ValueError,OverflowError) as e:raise ValueError('E_STRATO_TAU: invalid scalar') from e
    if not np.isfinite(tau) or tau<0:raise ValueError('E_STRATO_TAU: finite nonnegative required')
    return tau


def profile_config(state):
    cfg=state.get('stratosphere',{})
    if cfg.get('profile','gaussian')!='gaussian':raise ValueError('E_STRATO_PROFILE')
    try:center=float(cfg.get('center_km',21.));sigma=float(cfg.get('sigma_km',3.))
    except (TypeError,ValueError,OverflowError) as e:raise ValueError('E_STRATO_PROFILE_PARAMETERS') from e
    if not np.isfinite([center,sigma]).all():
        raise ValueError('E_STRATO_PROFILE_PARAMETERS')
    if sigma<1 or not 0<=center<=80:raise ValueError('E_STRATO_PROFILE_UNSUPPORTED')
    # Fixed trapezoid plus resolution derived from sigma; numerical normalization.
    resolution=80/sigma*40
    if not np.isfinite(resolution) or resolution>1_000_000:raise ValueError('E_STRATO_PROFILE_RESOLUTION')
    count=max(8001,int(math.ceil(resolution))+1)
    if count>1_000_001:raise ValueError('E_STRATO_PROFILE_RESOLUTION')
    z=np.linspace(0,80,count);raw=np.exp(-.5*((z-center)/sigma)**2)
    norm=float(np.trapezoid(raw,z))
    return dict(center=center,sigma=sigma,norm=norm,normalization_nodes=count)


def density(h,profile):
    h=np.asarray(h)
    return np.exp(-.5*((h-profile['center'])/profile['sigma'])**2)/profile['norm']


def cell_density(h,profile):
    """Voronoi r-node cell average: integrate q exactly, preserve the column.

    Gaussian has infinite tails: supported layer means its center +/-3 sigma
    fits within the grid. First-order and extinction still use analytic q.
    """
    h=np.asarray(h,float)
    if h.ndim!=1 or len(h)<2 or not np.isfinite(h).all() or np.any(np.diff(h)<=0):
        raise ValueError('E_STRATO_PROFILE_UNSUPPORTED')
    center,sigma=profile['center'],profile['sigma']
    if center-3*sigma<h[0] or center+3*sigma>h[-1]:raise ValueError('E_STRATO_PROFILE_UNSUPPORTED')
    edges=np.r_[h[0],(h[:-1]+h[1:])/2,h[-1]]
    cdf=np.array([math.erf((z-center)/(math.sqrt(2)*sigma)) for z in edges])
    average=np.diff(cdf)*sigma*math.sqrt(math.pi/2)/profile['norm']/np.diff(edges)
    if abs(float(average@np.diff(edges))-1)>1e-6:raise ValueError('E_STRATO_PROFILE_UNSUPPORTED')
    return average,edges


def table_path(state):
    raw=state.get('stratosphere',{}).get('optics_table');path=Path(raw) if raw else DEFAULT
    return path if path.is_absolute() else Path(__file__).resolve().parent/path   # relative tables resolve inside the package


def path_label(path):
    from . import paths
    return paths.rel(path) if Path(path).resolve().is_relative_to(paths.PKG) else Path(path).name


def computed_metadata(state,optics):
    """One metadata format shared by primary and derived runner paths."""
    computed=dict(state['parameters']);computed['stratosphere_tau550']=0. if optics is None else optics['meta']['stratosphere_tau550']
    raw=state.get('stratosphere',{});layer=dict(raw) if isinstance(raw,dict) else {};layer['status']='disabled' if optics is None else 'enabled'
    if optics is None:
        layer.pop('optics_sha256',None);layer.pop('optics_table',None)
    else:layer.update(optics_table=optics['meta']['optics_table'],optics_sha256=optics['meta']['optics_sha256'],polarization='fully depolarizing')
    return dict(parameters_requested=dict(state['parameters']),parameters_computed=computed,stratosphere=layer,
                stratosphere_computed=None if optics is None else dict(optics['meta']))


def load_table(path):
    try:d=json.loads(Path(path).read_text())
    except (OSError,ValueError) as e:raise ValueError('E_STRATO_TABLE_READ') from e
    try:
        rows=d['rows'];wl=np.array(d['wavelengths_nm'],float)
        ext=np.array([r['C_ext_um2'] for r in rows],float);sca=np.array([r['C_sca_um2'] for r in rows],float)
        ssa=np.array([r['SSA'] for r in rows],float);chi=np.array([r['chi_l'] for r in rows],float)
        g=np.array([r['g'] for r in rows],float);rwl=np.array([r['wavelength_nm'] for r in rows],float)
        if any(v.shape!=(len(wl),) for v in (ext,sca,ssa,g,rwl)):raise ValueError()
        if d['version']!=1 or wl.ndim!=1 or len(wl)<2 or chi.shape!=(len(wl),33) or not np.array_equal(wl,rwl):raise ValueError()
        if not all(np.isfinite(v).all() for v in (wl,ext,sca,ssa,chi,g)):raise ValueError()
        if np.any(wl<=0) or np.any(np.diff(wl)<=0) or np.any(ext<=0) or np.any(sca<0) or np.any(sca>ext*(1+1e-10)):raise ValueError()
        if np.any(ssa<0) or np.any(ssa>1+1e-10) or np.max(abs(ssa-sca/ext))>1e-10:raise ValueError()
        if np.max(abs(chi[:,0]-1))>1e-8 or np.max(abs(g-chi[:,1]/3))>1e-8:raise ValueError()
        if np.any(abs(chi)>2*np.arange(33)+1+1e-8) or not wl[0]<=550<=wl[-1]:raise ValueError()
    except (KeyError,TypeError,ValueError,IndexError) as e:raise ValueError('E_STRATO_TABLE_SCHEMA') from e
    ssa=np.where(abs(ssa-1)<1e-10,1.,ssa);chi[:,0]=1.
    return wl,ext,ssa,chi


def build_optics(state,h_km,wavelengths_nm):
    tau=tau_of(state)
    if tau==0:return None  # Deliberately do not read/validate unused layer metadata.
    profile=profile_config(state);path=table_path(state)
    source_den,cell_edges=cell_density(h_km,profile)
    wl,ce,ssa,chi=load_table(path);wanted=np.asarray(wavelengths_nm,float)
    if wanted.ndim!=1 or not np.isfinite(wanted).all() or np.any(wanted<wl[0]) or np.any(wanted>wl[-1]):raise ValueError('E_STRATO_WAVELENGTH_RANGE')
    ext=tau*np.interp(wanted,wl,ce)/np.interp(550.,wl,ce)
    scatter=ext*np.interp(wanted,wl,ssa);coeff=np.stack([np.interp(wanted,wl,chi[:,l]) for l in range(33)])
    den=density(np.asarray(h_km,float),profile)
    return dict(profile=profile,ext=ext,sca=scatter,density=den,source_density=source_den,cell_edges=cell_edges,beta_e=den[:,None]*ext,beta_s=den[:,None]*scatter,
                chi=coeff,eigen=coeff/(2*np.arange(33)[:,None]+1),
                meta=dict(stratosphere_tau550=tau,aod550_semantics='troposphere only',profile='gaussian',center_km=profile['center'],
                          sigma_km=profile['sigma'],normalization_km=profile['norm'],normalization_nodes=profile['normalization_nodes'],
                          status='enabled',source_density='column-conserving r-cell average; first/extinction analytic',
                          optics_table=path_label(path),optics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          phase='Mie scalar Legendre L32; no HG fallback',polarization='fully depolarizing',
                          particle_model='single lognormal sulfate population',source_stride=1132))


def _column_nodes(p,d,old,n):
    """geometry_segments' exact nodes/XYZ reduction, without unused optical prefixes.

    SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS (default on, '0' = off) sums the radius
    components separately, avoiding the [ray,node,xyz] temporary; was byte-identical
    to the q*q reduction (strato-kernel-loop cycle15, test_wall_host NOPREFIX bytes).
    """
    edges=old.probe.boundaries(p,d);end=edges[:,-1]
    b=np.sum(p*d,-1);pp=np.sum(p*p,-1);more=[]
    for height in (10.,25.,40.):
        disc=b*b-pp+(6371.+height)**2
        for sign in (-1,1):
            root=-b+sign*np.sqrt(np.maximum(disc,0))
            more.append(np.where((disc>0)&(root>0)&(root<end),root,end))
    edges=np.sort(np.column_stack([edges,*more]),axis=-1)
    u,wu=old.probe.gl(n)
    for k in range(edges.shape[1]-1):
        active=np.flatnonzero(edges[:,k+1]-edges[:,k]>1e-8)
        if not len(active):continue
        a,b=edges[active,k,None],edges[active,k+1,None]
        pa,da=p[active],d[active]
        ra=np.sum((pa+a*da)**2,-1);rb=np.sum((pa+b*da)**2,-1)
        lower=(ra<=rb)[:,None]
        t=np.where(lower,a+(b-a)*u*u,b-(b-a)*(1-u)**2)
        ds=np.where(lower,2*(b-a)*u*wu,2*(b-a)*(1-u)*wu)
        if os.environ.get('SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS','1')!='0':  # default on: measured bit-identical, ~2x in cycle15
            radius2=np.square(pa[:,None,0]+t*da[:,None,0])
            for axis in (1,2):
                radius2+=np.square(pa[:,None,axis]+t*da[:,None,axis])
            r=np.sqrt(radius2)
        else:
            q=pa[:,None,:]+t[...,None]*da[:,None,:]
            r=np.sqrt(np.sum(q*q,-1))
        yield active,r,ds


def columns(p,d,old,profile,n=24):
    """Same accepted segment decomposition and GL order as old total_columns."""
    p=np.broadcast_to(p,np.shape(d));out=np.zeros(len(d))
    if os.environ.get('SOSFUSE2_STRATO_HOST_NOPREFIX')=='1':
        for active,r,ds in _column_nodes(p,d,old,n):
            out[active]+=np.sum(density(r-6371.,profile)*ds,axis=1)
        return out
    for active,q,r,ds,prefix in old.geometry_segments(p,d,n):
        out[active]+=np.sum(density(r-6371.,profile)*ds,axis=1)
    return out


def solar_grid(s,old,profile):
    sun=s.geo.sun;out=np.empty(sun.col.shape[:2])
    for i,h in enumerate(sun.h):
        r=6371.+h;mg=-np.sqrt(max(0.,1-(6371./r)**2));mu=mg+(1-mg)*sun.x**2;mu[0]=mg+1e-9
        p=np.column_stack([np.zeros(len(mu)),np.zeros(len(mu)),np.full(len(mu),r)])
        d=np.column_stack([np.sqrt(np.maximum(1-mu*mu,0)),np.zeros(len(mu)),mu])
        out[i]=columns(p,d,old,profile,32)
    return out


def sun_lookup(q,r,sun,grid):
    h=np.clip(r-6371.,0,80);mu=q[...,2]/r;mg=-np.sqrt(np.maximum(0,1-(6371./np.maximum(r,6371.))**2))
    x=np.sqrt(np.clip((mu-mg)/(1-mg),0,1));i=np.clip(np.searchsorted(sun.h,h,side='right')-1,0,len(sun.h)-2)
    a=(h-sun.h[i])/(sun.h[i+1]-sun.h[i]);f=x*(len(sun.x)-1);j=np.clip(f.astype(int),0,len(sun.x)-2);b=f-j
    col=(1-a)*(1-b)*grid[i,j]+a*(1-b)*grid[i+1,j]+(1-a)*b*grid[i,j+1]+a*b*grid[i+1,j+1]
    return col,mu>=mg


GEOM_FIELDS=('loc','w','sc','lit','col','st','ssc','ec','re')
_GEOM_MEMO={}


def _geometry_key(s,scope,profile):
    """Everything the per-cell ground geometry below reads, except the state optics combined afterwards.
    Unknown dependence is fail-safe only if it shows in these inputs; verified by array equality (2026-10-03)."""
    from .density import profile_identity
    h=hashlib.sha256()
    for a in (s.geo.h,s.geo.chi,s.geo.sun.h,s.geo.sun.x,s.geo.sun.col):h.update(np.ascontiguousarray(a).tobytes())
    # the code that computes col/st/geom, so an edit to it cannot reuse a stale npz.
    here=Path(__file__).resolve().parent
    for name in ('stratosphere.py','ground_parallel.py','ray_columns.py','density.py','column_table.py','column_parallel.py'):
        h.update(name.encode());h.update(hashlib.sha256((here/name).read_bytes()).digest())
    d=getattr(s,'density_optics',None)
    p={k:v for k,v in s.fuse_state.get('parameters',{}).items() if k not in ATTACH_OPTICS_PARAMS}
    # ②b: the PROFILE identity, not a coincidental sun.col byte match; k3 #12: the ground rays' PROFILE method likewise.
    h.update(json.dumps(dict(profile=[profile['center'],profile['sigma'],profile['norm']],cfg=s.geo.cfg,params=p,
        density=None if d is None else [d['meta'].get('asset_sha256'),d['mask'],profile_identity(d)['sha256'],_sun_method(s)]),
        sort_keys=True,default=str).encode())
    return h.hexdigest()[:24]


def _sun_method(s):
    """The PROFILE method density.attach resolved for this solver's sun table (density.sun_method when attach did not run)."""
    from .density import sun_method
    return getattr(s,'density_sun_method',None) or sun_method()


def _ground_geometry(s,scope,old,profile):
    """(suncol, per-cell [loc,w,sc,lit,col,st,ssc,ec,re]): the state-independent, expensive part of attach.
    Always memoized in-process by _geometry_key (wavelength-independent: one state's FW15 bundles share it); STRATO_ATTACH_CACHE=<dir>
    adds the on-disk copy. Before 2026-11-03 the memo was tied to the directory, so a bake with the cache off recomputed it for every
    bundle (L4: 23.6-29.1 s each)."""
    import os
    root=os.environ.get('STRATO_ATTACH_CACHE','').strip()
    key=_geometry_key(s,scope,profile)
    if key in _GEOM_MEMO:return _GEOM_MEMO[key]
    path=Path(root)/f'attach_geometry_{key}.npz' if root else None
    if path is not None and path.exists():
        with np.load(path,allow_pickle=False) as z:
            off=z['offsets'];suncol=z['suncol'];cat={f:z[f] for f in GEOM_FIELDS}
        geom=[tuple(cat[f][off[i]:off[i+1]] for f in GEOM_FIELDS) for i in range(len(off)-1)]
        _GEOM_MEMO[key]=(suncol,geom);return suncol,geom
    from .ground_parallel import cells
    suncol=solar_grid(s,old,profile)
    geom=cells(s,scope,'strato',getattr(s,'density_optics',None),profile,suncol,method=_sun_method(s))
    _GEOM_MEMO[key]=(suncol,geom)
    if path is not None:
        off=np.r_[0,np.cumsum([len(g[0]) for g in geom])].astype('i8')
        tmp=path.with_suffix('.tmp.npz');Path(root).mkdir(parents=True,exist_ok=True)
        np.savez(tmp,offsets=off,suncol=suncol,**{f:np.concatenate([g[i] for g in geom]) for i,f in enumerate(GEOM_FIELDS)})
        os.replace(tmp,path)
    return suncol,geom


def attach(s,scope):
    """Call once after fuse_state is assigned, before kernels/ground are used.

    The accepted solver owns immutable state during a solve. Reassigning state
    on a live solver is unsupported; construct a new solver for each state.
    """
    if hasattr(s,'strato'):return s.strato
    optics=build_optics(s.fuse_state,s.geo.h,s.wl)
    if optics is None:return None
    from .sh_layout import count
    optics['meta']['source_stride']=10+2*count(getattr(s.kernel,'L',32))
    old=scope['b'].sh.old;profile=optics['profile']
    suncol,geom=_ground_geometry(s,scope,old,profile)
    # Recompute first-ground values with additional sun->ground and ground->cell
    # extinction. Preserve accepted angular rule and Rayleigh tensor contraction.
    ds_all=[];vs=[];off=[0];gm=np.zeros((s.geo.ns,4,s.W),'f4')
    for cell in range(s.geo.ns):
        loc,w,sc,lit,col,st,ssc,ec,re=geom[cell]
        od=(sc+col)@s.ext+(st+ssc)[:,None]*optics['ext']
        value=s.fuse_state['parameters']['albedo']/np.pi*np.maximum(0,np.cos(ec))[:,None]*np.exp(-od)*lit[:,None]*(re<6371.+1e-5)[:,None]*w[:,None]
        take=np.any(value>0,axis=1);ds_all.append(loc[take]);vs.append(value[take]);off.append(off[-1]+take.sum())
        factor=np.stack([(1-loc[:,0]**2)/2,(1-loc[:,1]**2)/2,(1-loc[:,2]**2)/2,-loc[:,0]*loc[:,2]/2],-1)
        gm[cell]=(factor.T@value)*(3/(8*np.pi)*s.ext[0])[None,:]
    # Commit only after successful optical preparation, so failed setup is retryable.
    optics.update(sun_col=suncol,old=old)
    s.goff=np.array(off,'i8');s.gdir=np.ascontiguousarray(np.concatenate(ds_all),'f4');s.gval=np.ascontiguousarray(np.concatenate(vs),'f4');s.gm=gm
    s.strato=optics
    return optics


def source_size(s):
    from .sh_layout import source_size as size
    return size(s)


def eigen_by_sh(optics, L=32):
    from .sh_layout import count
    count(L)
    levels=np.array([l for l in range(L+1) for m in range(l+1)])
    return optics['eigen'][levels]
