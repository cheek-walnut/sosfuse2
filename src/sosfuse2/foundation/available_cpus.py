#!/usr/bin/env python3
"""Cores this process may use. Unlike `nproc`, this reads scheduler affinity and cgroup quota without applying
OMP_NUM_THREADS. Prints one integer, or every source read with --explain."""

import math, os, subprocess, sys


def cores():
    seen = {}
    try:
        seen['affinity'] = len(os.sched_getaffinity(0))
    except AttributeError:
        seen['cpu_count'] = os.cpu_count() or 1
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if quota != 'max':
            seen['cgroup'] = max(1, math.floor(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    env = dict(os.environ)
    env.pop('OMP_NUM_THREADS', None)
    try:
        seen['nproc_without_omp'] = int(subprocess.check_output(['nproc'], env=env, text=True))
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    if 'OMP_NUM_THREADS' in os.environ:
        seen['note_OMP_NUM_THREADS_set_to'] = os.environ['OMP_NUM_THREADS']
    return min(v for k, v in seen.items() if not k.startswith('note')), seen


if __name__ == '__main__':
    n, seen = cores()
    print(n)
    if '--explain' in sys.argv:
        print(seen, file=sys.stderr)
