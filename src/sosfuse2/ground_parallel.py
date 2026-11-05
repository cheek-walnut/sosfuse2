"""Spawn whole ground cells: preserve incoming quadrature and each cell's reduction order."""
import json
import multiprocessing
import os
import time
import types
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .column_parallel import default_workers, event_fields, freeze, thaw

ENV = 'SOSFUSE2_GROUND_WORKERS'
_STATE = None


def workers():
    raw = os.environ.get(ENV, '').strip()
    if not raw:return default_workers()
    try:
        value = int(raw)
        if value > 0:return value
    except ValueError:pass
    raise ValueError('E_GROUND_WORKERS: expected positive integer')


def _functions(scope, sun):
    old = scope['b'].sh.old
    incoming = scope['b'].sh.incoming
    # incoming's exec scope contains the entire solver. Ship only the GL dependency.
    globals_ = dict(incoming.__globals__, old=types.SimpleNamespace(
        probe=types.SimpleNamespace(gl=old.probe.gl)))
    incoming = types.FunctionType(incoming.__code__, globals_, incoming.__name__, incoming.__defaults__)
    return types.SimpleNamespace(incoming=incoming, frame=old.probe.frame,
        boundaries=old.probe.boundaries, gl=old.probe.gl, total_columns=old.total_columns,
        geometry_segments=old.geometry_segments, profiles=old.profiles, lookup=type(sun).lookup)


def _init(frozen, data, kind):
    global _STATE
    f = thaw(frozen)
    f.probe = types.SimpleNamespace(boundaries=f.boundaries, gl=f.gl)
    _STATE = (f, data, kind)


def _run(task):
    return _cell(*_STATE, task)


def _cell(f, data, kind, task):
    if kind == 'aerosol':
        p, d = task
        return f.total_columns(p, d, 24)[:,1]
    h, chi, radius, sun, optics, profile, suncol, method = data
    cell = task;ih, ic = divmod(cell, len(chi));r = radius+h[ih];ch = chi[ic]
    loc, w = f.incoming(r, ch, 1)
    p, up, ex = f.frame(np.array(r), ch);ey = np.cross(up, ex)
    d = loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p = np.broadcast_to(p,d.shape)
    end = f.boundaries(p,d)[:,-1];q = p+end[:,None]*d;re = np.linalg.norm(q,axis=-1)
    from .ray_columns import integrate, replace
    exact = optics is not None and method == 'exact'
    # exact (density.sun_method, T3g): the GL24 walk keeps the base and strato columns; the selected PROFILE columns
    # come from the closed form over the same [0, end] (host and strato kinds alike).
    col, corrected, st = integrate(p,d,f,24,legacy=True,optics=None if exact else optics,
                                   profile=profile if kind == 'strato' else None)
    if exact:
        from .density import exact_profile_columns
        corrected = exact_profile_columns(p,d,end,optics)
    if optics is not None:col = replace(col,corrected,optics)
    sc, lit = f.lookup(sun,q,np.maximum(radius,re))
    ec = np.arccos(np.clip(q[:,2]/re,-1,1))
    if kind == 'host':
        candidate = (np.cos(ec)>0)&lit&(re<radius+1e-5)&(w>0)
        return loc,w,q,re,col[:,[0,2]],candidate
    from .stratosphere import sun_lookup
    ssc,_ = sun_lookup(q,np.maximum(radius,re),sun,suncol)
    return loc,w,sc,lit,col,st,ssc,ec,re


def _map(f, data, kind, tasks):
    count = workers();started = time.monotonic()
    if count > 1 and len(tasks) > 1:
        result, reason = 'parallel', None
        try:
            frozen = freeze(f, list(vars(f)))
            with ProcessPoolExecutor(min(count,len(tasks)), mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init, initargs=(frozen,data,kind)) as pool:
                out = list(pool.map(_run,tasks))
        except Exception as exc:
            result = 'fallback'
            reason = str(exc).split(':')[0] if str(exc).startswith('E_') else type(exc).__name__
        print(json.dumps(dict(event='PHASE',phase='ground_parallel',**event_fields(),kind=kind,workers=count,
                              result=result,reason=reason,seconds=time.monotonic()-started)),flush=True)
        if result == 'parallel':return out
    # No partial writes escape a failed map.
    f.probe = types.SimpleNamespace(boundaries=f.boundaries,gl=f.gl)
    return [_cell(f,data,kind,task) for task in tasks]


def cells(s, scope, kind, optics=None, profile=None, suncol=None, method=None):
    """method: density.sun_method() value for the PROFILE columns (resolved here when omitted), shipped to every worker."""
    from .density import SUN_METHODS, sun_method
    method = 'gl' if optics is None else sun_method() if method is None else method
    if method not in SUN_METHODS:raise ValueError('E_SUN_METHOD: expected gl or exact')
    g = s.geo
    sun = types.SimpleNamespace(h=g.sun.h,x=g.sun.x,col=g.sun.col)
    optics = None if optics is None else {k:v for k,v in optics.items() if k != 'old'}
    data = (g.h,g.chi,6371. if kind == 'strato' else scope['b'].R,sun,optics,profile,suncol,method)
    return _map(_functions(scope,g.sun),data,kind,list(range(g.ns)))


def aerosol(scope, sun, p, d):
    if not len(d):return np.empty(0)
    tasks = [(p[lo:lo+8192],d[lo:lo+8192]) for lo in range(0,len(d),8192)]
    return np.concatenate(_map(_functions(scope,sun),None,'aerosol',tasks))
