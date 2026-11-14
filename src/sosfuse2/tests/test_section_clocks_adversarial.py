"""Independent frozen-matrix instrumentation checks; no CUDA numerical claim.

Set TMPDIR to a scratch folder, then run python -B -m sosfuse2.tests.test_section_clocks_adversarial.
The fake CUDA seam exercises real launch plumbing, not device arithmetic.
"""

import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.foundation import paths
from sosfuse2.checks.numerical_checks import assert_equivalent
from sosfuse2.solve import source_solver
from sosfuse2.checks import testkit


def evidence():
    """The kernel_ab rig evidence pack (sky-sos record; skipped without SOSFUSE2_SKY_SOS_ROOT, failed when it is set and missing)."""
    return testkit.sky_sos('rig/docker/received_pack/kernel_ab')


def campaign_checks():
    return testkit.sky_sos('rig/campaign_usstd/campaign_checks.py')


FLAG = 'EXP_SECTION_CLOCKS'
A = ('LANE_GROUND', 'LM_LUT', 'PHASE2_NOBARRIER', 'SH_AUX', 'SPECIALIZE')
NAMES = ('point', 'ground', 'sh_basis', 'transport', 'attenuation', 'projection', 'barrier', 'total')


def baseline_module():
    path = evidence() / 'prechange_source/fused.py'
    spec = importlib.util.spec_from_file_location('sosfuse2._clock_baseline', path)
    module = importlib.util.module_from_spec(spec)
    with testkit.legacy_module_aliases('reference', 'arithmetic', 'timers'):
        spec.loader.exec_module(module)
    return module


class FakeXP:
    """Separate, observable allocation/event seam backed by NumPy arrays."""

    def __init__(self):
        self.allocations = []
        self.events = []
        self.cuda = SimpleNamespace(Event=self.event, get_elapsed_time=lambda a, b: 12.5)

    def event(self):
        event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
        self.events.append(event)
        return event

    def empty(self, shape, dtype):
        return np.full(shape, -123, dtype=dtype)

    def zeros(self, shape, dtype):
        value = np.zeros(shape, dtype=dtype)
        self.allocations.append(value)
        return value

    ascontiguousarray = staticmethod(np.ascontiguousarray)
    transpose = staticmethod(np.transpose)
    diff = staticmethod(np.diff)
    argsort = staticmethod(np.argsort)
    asnumpy = staticmethod(np.asarray)


class FakeRaw:
    """Records numeric ABI and writes distinguishable counter slots only."""

    def __init__(self, clocks):
        self.clocks = clocks
        self.calls = []
        self.counter_views = []

    def __call__(self, grid, block, args, shared_mem):
        self.calls.append((grid, block, args, shared_mem))
        c0 = int(args[10][0, 0])  # public starts[cell,xyz] is offset by the wrapper
        # Explicit kernel ABI: 9 scalar + 12 geometry + 17 constants + 9 tail.
        moments, flux, query = args[44:47]
        phase = int(args[7])
        if phase == 0:
            for c in range(grid[0]):
                moments[c] = 1000 + c0 + c
        else:
            moments.fill(0)
        for c in range(grid[0]):
            flux[c] = 2000 + c0 + c
        query.fill(3000)
        if self.clocks:
            counters = args[-1]
            assert counters.dtype == np.uint64
            self.counter_views.append(counters)
            for c, tile, warp in itertools.product(range(grid[0]), range(grid[1]), range(block[0] // 32)):
                marker = 1 + (c0 + c) * 100 + tile * 10 + warp
                values = [marker + i for i in range(7)]
                counters[c, tile, warp] = values + [sum(values) + 3]


def launch_fixture(module=source_solver, flags=(), W=15, cells=5, block=256):
    kernel = module.Kernel(flags=flags, block_size=block)
    xp = FakeXP()
    kernel.xp = xp
    raw = FakeRaw(FLAG in flags)
    kernel.raw_kernel = lambda *a, **k: raw
    kernel.sync = lambda: None
    kernel.memory = lambda: None
    state = SimpleNamespace(W=W, geo=SimpleNamespace(h=np.zeros(2), chi=np.zeros(2)))
    constants = dict(
        height=np.float32(1),
        g=np.float32(0.8),
        mix=np.float32(0.5),
        albedo=0.2,
        sun_h=np.zeros(2),
        sun_col=np.zeros((2, 2)),
    )
    kernel.setup = lambda *a: constants
    dc = {
        name: np.arange(3, dtype='f4')
        for name in (
            'u',
            'wu',
            'aa',
            'bb',
            'dd',
            'h',
            'chi',
            'hr',
            'ha',
            'ext',
            'asca',
            'sun_h',
            'sun_col',
            'gm',
            'goff',
            'gdir',
            'gval',
        )
    }
    kernel.constants_device = lambda *a: dc
    names = (
        'cell_off',
        'starts',
        'targets',
        'dirs',
        'weights',
        'edge_off',
        'edges',
        'lower',
        'end_j',
        'end_f',
        'end_col',
        'hits',
    )
    arrays = {name: np.arange(cells + 1) for name in names}
    arrays['starts'] = np.stack((np.arange(cells), np.zeros(cells), np.zeros(cells)), axis=1)
    arrays['targets'] = np.zeros((cells, 9))
    bundle = SimpleNamespace(nq=2, cells=cells, rays=cells, arrays=arrays)
    field = np.arange(cells * 571 * W, dtype='f4').reshape(cells, 571, W)
    return kernel, xp, raw, state, bundle, field


def run_launch(fixture, directory, *, module=source_solver, chunk=0, phase=0, first=False, timed=False):
    kernel, xp, raw, state, bundle, field = fixture
    text = io.StringIO()
    with (
        patch.dict(
            os.environ,
            {
                'SOSFUSE2_SECTION_CLOCKS_DIR': str(directory),
                'SOSFUSE2_LAUNCH_CELLS': str(chunk),
                'CUDA_VISIBLE_DEVICES': '1',
            },
        ),
        patch.object(module, 'stage', lambda *a, **k: contextlib.nullcontext()),
        contextlib.redirect_stdout(text),
    ):
        output = kernel.launch(bundle, state, field, first=first, phase=phase, timed=timed)
    reports = [
        json.loads(line.split(' ', 1)[1]) for line in text.getvalue().splitlines() if line.startswith('SECTION_CLOCKS ')
    ]
    return output, reports


class SummaryAdversarial(unittest.TestCase):
    def test_exact_schema_names(self):
        self.assertEqual(source_solver.SECTION_CLOCK_NAMES, NAMES)

    def test_large_uint64_sums_are_lossless_and_not_wall_time(self):
        counters = np.zeros((2, 3, 8, 8), dtype='uint64')
        counters[..., 0] = 2**63 + 17
        counters[..., 7] = 2**63 + 39
        report = source_solver.section_clock_summary(counters)
        self.assertEqual(report['observer_cycles']['point'], 48 * (2**63 + 17))
        self.assertEqual(report['observer_cycles']['total'], 48 * (2**63 + 39))
        self.assertEqual(report['residual_observer_cycles'], 48 * 22)
        self.assertIsInstance(json.loads(json.dumps(report))['observer_cycles']['point'], int)
        self.assertIn('not wall time', report['accounting'])
        self.assertIn('not pure waiting', report['accounting'])
        self.assertFalse(any(key in report for key in ('seconds', 'wall_seconds')))

    def test_sum_of_several_large_sections_does_not_wrap(self):
        counters = np.zeros((1, 1, 1, 8), dtype='uint64')
        counters[..., :2] = 2**63
        counters[..., 7] = np.iinfo(np.uint64).max
        with self.assertRaisesRegex(ValueError, 'E_SECTION_CLOCK_OVERLAP'):
            source_solver.section_clock_summary(counters)

    def test_per_observer_overlap_cannot_hide_in_aggregate(self):
        counters = np.zeros((1, 1, 2, 8), dtype='uint64')
        counters[0, 0, 0, :2] = (11, 10)
        counters[0, 0, 0, 7] = 20
        counters[0, 0, 1, 7] = 1000
        with self.assertRaisesRegex(ValueError, 'E_SECTION_CLOCK_OVERLAP'):
            source_solver.section_clock_summary(counters)

    def test_disjoint_sections_residual_and_zero_total(self):
        counters = np.zeros((1, 1, 1, 8), dtype='uint64')
        zero = source_solver.section_clock_summary(counters)
        self.assertTrue(all(value is None for value in zero['fractions'].values()))
        counters[0, 0, 0] = [1, 2, 3, 4, 5, 6, 7, 30]
        report = source_solver.section_clock_summary(counters)
        self.assertEqual(report['residual_observer_cycles'], 2)
        self.assertAlmostEqual(sum(report['fractions'].values()), 28 / 30)

    def test_invalid_schema_fails_distinguishably(self):
        for value in (
            np.zeros((1, 1, 1, 8), 'int64'),
            np.zeros((1, 1, 1, 8), 'float64'),
            np.zeros((1, 1, 8), 'uint64'),
            np.zeros((1, 1, 1, 7), 'uint64'),
            np.zeros((0, 1, 1, 8), 'uint64'),
        ):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaisesRegex(ValueError, 'E_SECTION_CLOCK_BUFFER'):
                    source_solver.section_clock_summary(value)

    def test_noncontiguous_schema_is_summed_without_mutating_input(self):
        counters = np.zeros((2, 3, 8, 8), dtype='uint64')[:, ::2]
        counters[..., 0] = 3
        counters[..., 7] = 5
        before = counters.copy()
        report = source_solver.section_clock_summary(counters)
        self.assertEqual(report['observer_cycles']['point'], 2 * 2 * 8 * 3)
        np.testing.assert_array_equal(counters, before)


class FlagAdversarial(unittest.TestCase):
    def test_explicit_environment_token_existing_grammar(self):
        with patch.dict(os.environ, {'SOSFUSE2_KERNEL_FLAGS': ' SH_AUX, EXP_SECTION_CLOCKS,, '}):
            kernel = source_solver.Kernel()
        self.assertEqual(kernel.flags, (FLAG,))

    def test_lookalikes_are_rejected_and_never_enabled(self):
        for token in (
            'exp_section_clocks',
            'EXP_SECTION_CLOCKS_EXTRA',
            'PRE_EXP_SECTION_CLOCKS',
            'XEXP_SECTION_CLOCKSX',
        ):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, 'E_KERNEL_FLAG'):
                    source_solver.Kernel(flags=(token,))

    def test_empty_explicit_flags_override_environment_and_duplicates_are_idempotent(self):
        with patch.dict(os.environ, {'SOSFUSE2_KERNEL_FLAGS': FLAG}):
            self.assertEqual(source_solver.Kernel(flags=()).flags, ())
            self.assertEqual(source_solver.Kernel(flags=(FLAG, FLAG)).flags, (FLAG,))

    def test_compile_options_and_cache_isolate_clock_flag_and_tile_stage(self):
        compiled = []

        class FakeCompiled:
            def __init__(self, source, name, options, backend):
                self.options = options
                self.attributes = {}
                compiled.append(self)

            def compile(self):
                pass

        xp = FakeXP()
        xp.RawKernel = FakeCompiled
        xp.cuda.runtime = SimpleNamespace(
            getDevice=lambda: 0, getDeviceProperties=lambda device: {'sharedMemPerBlock': 200000}
        )
        flags = A + ('LANE_ATTEN',)  # Current Simpson path; keep immutable baseline A/B unchanged.
        kernel = source_solver.Kernel(flags=flags)
        kernel.xp = xp
        with patch.object(source_solver, 'stage', lambda *a, **k: contextlib.nullcontext()):
            off_first = kernel.raw_kernel(15, first=1, phase=0)
            self.assertNotIn('-D' + FLAG, off_first.options)
            kernel.flags = tuple(sorted(flags + (FLAG,)))
            on_first = kernel.raw_kernel(15, first=1, phase=0)
            self.assertIn('-D' + FLAG, on_first.options)
            self.assertIsNot(on_first, off_first)
            on_apply = kernel.raw_kernel(15, first=0, phase=0)
            self.assertIn('-D' + FLAG, on_apply.options)
            self.assertIsNot(on_apply, on_first)
            kernel.flags = tuple(sorted(flags))
            self.assertIs(kernel.raw_kernel(15, first=1, phase=0), off_first)
        self.assertEqual(len(compiled), 3)


class LaunchAdversarial(unittest.TestCase):
    def test_missing_clock_directory_fails_before_numeric_launch(self):
        fixture = launch_fixture(flags=(FLAG,))
        kernel, xp, raw, state, bundle, field = fixture
        with patch.dict(os.environ, {'SOSFUSE2_SECTION_CLOCKS_DIR': ''}):
            with self.assertRaisesRegex(ValueError, 'E_SECTION_CLOCK_OUTPUT'):
                kernel.launch(bundle, state, field)
        self.assertEqual(raw.calls, [])
        self.assertEqual(xp.allocations, [])

    def test_disabled_abi_matches_original_default_A_B_and_chunking(self):
        original = baseline_module()
        with tempfile.TemporaryDirectory() as directory:
            for flags, chunk in itertools.product(((), A), (0, 2)):
                with self.subTest(flags=flags, chunk=chunk):
                    before = launch_fixture(original, flags)
                    after = launch_fixture(source_solver, flags)
                    bo, br = run_launch(before, Path(directory) / 'before', module=original, chunk=chunk)
                    ao, ar = run_launch(after, Path(directory) / 'after', chunk=chunk)
                    self.assertEqual((br, ar), ([], []))
                    self.assertFalse((Path(directory) / 'after').exists())
                    self.assertEqual(after[1].events, [])
                    for left, right in zip(bo, ao):
                        assert_equivalent(left, right)
                    self.assertEqual(len(before[2].calls), len(after[2].calls))
                    for bcall, acall in zip(before[2].calls, after[2].calls):
                        self.assertEqual(bcall[:2], acall[:2])
                        self.assertEqual(bcall[3], acall[3])
                        self.assertEqual(len(bcall[2]), len(acall[2]))
                        for bvalue, avalue in zip(bcall[2], acall[2]):
                            if isinstance(bvalue, np.ndarray):
                                self.assertEqual(bvalue.dtype, avalue.dtype)
                                self.assertEqual(bvalue.shape, avalue.shape)
                                assert_equivalent(bvalue, avalue)
                            else:
                                self.assertEqual(bvalue, avalue)

    def test_layout_tile_block_chunk_cross_product_and_numeric_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = itertools.product(((), A), (1, 15), (32, 256), (0, 2), (0, 1, 2))
            for flags, W, block, chunk, phase in cases:
                with self.subTest(flags=flags, W=W, block=block, chunk=chunk, phase=phase):
                    fixture = launch_fixture(flags=flags + (FLAG,), W=W, block=block)
                    kernel, xp, raw, state, bundle, field = fixture
                    field_before = field.tobytes()
                    output, reports = run_launch(fixture, directory, chunk=chunk, phase=phase)
                    self.assertEqual(len(reports), 1)
                    report = reports[0]
                    tile_count = 1
                    self.assertEqual(report['shape'], [5, tile_count, block // 32, 8])
                    self.assertEqual(report['grid'], [5, tile_count])
                    self.assertEqual(report['launch_cells'], chunk)
                    with np.load(report['raw_path'], allow_pickle=False) as data:
                        self.assertEqual(data.files, ['cycles'])
                        counters = data['cycles']
                    for c, tile, warp in itertools.product(range(5), range(tile_count), range(block // 32)):
                        marker = 1 + c * 100 + tile * 10 + warp
                        self.assertEqual(int(counters[c, tile, warp, 0]), marker)
                    self.assertEqual(report['residual_observer_cycles'], 5 * tile_count * (block // 32) * 3)
                    self.assertEqual(field.tobytes(), field_before)
                    counter_root = next(a for a in xp.allocations if a.dtype == np.uint64)
                    for view in raw.counter_views:
                        self.assertTrue(np.shares_memory(view, counter_root))
                        self.assertFalse(any(np.shares_memory(view, a) for a in output))
                        self.assertFalse(np.shares_memory(view, field))
                    if chunk:
                        self.assertEqual([call[0][0] for call in raw.calls], [2, 2, 1])
                        self.assertFalse(np.shares_memory(raw.counter_views[0], raw.counter_views[1]))

    def test_new_buffer_each_launch_raw_artifacts_provenance_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = launch_fixture(flags=A + (FLAG,))
            out1, reports1 = run_launch(fixture, directory)
            counters1 = fixture[2].counter_views[-1]
            counters1.fill(2**63)  # allocator/prior-launch poison cannot affect a fresh buffer
            out2, reports2 = run_launch(fixture, directory, timed=True)
            r1, r2 = reports1[0], reports2[0]
            self.assertFalse(np.shares_memory(counters1, fixture[2].counter_views[-1]))
            self.assertNotEqual(r1['raw_path'], r2['raw_path'])
            self.assertEqual([r1['launch'], r2['launch']], [0, 1])
            self.assertEqual(r1['observer_cycles'], r2['observer_cycles'])
            for report in (r1, r2):
                self.assertEqual(report['schema'], 'sosfuse2-section-clocks-v1')
                self.assertEqual(report['gpu'], '1')
                self.assertEqual(report['pid'], os.getpid())
                self.assertEqual(report['phase'], 0)
                self.assertFalse(report['first'])
                self.assertEqual(report['wavelengths'], 15)
                self.assertEqual(report['block_size'], 256)
                self.assertEqual(set(report['flags']), {FLAG})
                self.assertEqual(report['cuda_event_seconds'], 0.0125)
                self.assertTrue(Path(report['raw_path']).is_file())
            self.assertEqual(len(fixture[1].events), 4)
            self.assertEqual(fixture[0].cuda_events, [0.0125])
            for a, b in zip(out1, out2):
                assert_equivalent(a, b)

    def test_clock_on_off_on_does_not_leak_argument_or_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = launch_fixture(flags=A + (FLAG,))
            _, first = run_launch(fixture, directory)
            fixture[0].flags = tuple(sorted(A))
            fixture[2].clocks = False
            _, off = run_launch(fixture, directory)
            self.assertEqual(off, [])
            self.assertEqual(len(fixture[2].calls[-1][2]), len(fixture[2].calls[-2][2]) - 1)
            self.assertEqual(fixture[2].calls[-1][2][47].dtype, np.dtype('f4'))  # paired SH sidecar
            self.assertEqual(fixture[2].calls[-1][2][-1].dtype, np.dtype('i4'))  # cell-order tail
            fixture[0].flags = tuple(sorted(A + (FLAG,)))
            fixture[2].clocks = True
            _, last = run_launch(fixture, directory)
            self.assertEqual(first[0]['observer_cycles'], last[0]['observer_cycles'])


class IndependentEvidence(unittest.TestCase):
    def test_baseline_reference_hashes_and_judge(self):
        for name in ('A', 'B', 'default'):
            with self.subTest(name=name):
                path = evidence() / 'prechange_selftests' / name
                digest = hashlib.sha256((path / 'selftest_values.npz').read_bytes()).hexdigest()
                self.assertEqual(digest, 'b68be01c94f8f39225e545c2344623583eba394cb314bd5eebbd71d27fa6b049')
                self.assertTrue(json.loads((path / 'judge.json').read_text())['pass_'])

    def test_offflag_tokens_match_independent_clock_removal(self):
        """Observer-removal check only; numerical regression belongs to G-P/G-T.

        Strip entire observer-only blocks, including their off branches, and supply
        independent no-ops. This cannot detect numerical edits common to both copies.
        """
        source = paths.KERNEL_CU.read_text()

        def without_clocks(code):
            lines = code.splitlines(keepends=True)
            kept = []
            depth = 0
            for line in lines:
                if not depth and line.strip() == '#ifdef EXP_SECTION_CLOCKS':
                    depth = 1
                elif depth:
                    if re.match(r'\s*#\s*if(?:def|ndef)?\b', line):
                        depth += 1
                    elif re.match(r'\s*#\s*endif\b', line):
                        depth -= 1
                else:
                    kept.append(line)
            self.assertEqual(depth, 0)
            return (
                '#define SECTION_PARAM\n#define SECTION_BEGIN()\n#define SECTION_BEGIN_IF(c)\n'
                '#define SECTION_END(k)\n#define SECTION_END_IF(k,c)\n#define SECTION_SYNC(op) op\n' + ''.join(kept)
            )

        def pp(code, options):
            out = subprocess.check_output(['clang++', '-E', '-P', '-x', 'c++', *options, '-'], input=code, text=True)
            return re.findall(r'\w+|[^\s]', out)

        testkit.require_tool('clang++')
        reference = without_clocks(source)
        specs = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1, 2, 0), (0, 2, 0))
        for W, strato, spec in itertools.product((1, 15), (False, True), specs):
            with self.subTest(W=W, strato=strato, spec=spec):
                options = [f'-DFW={W}', f'-DK_FIRST={spec[0]}', f'-DK_PHASE={spec[1]}', f'-DK_BOUNDARY={spec[2]}']
                if strato:
                    options += ['-DSTRATO']
                off = pp(source, options)
                self.assertEqual(pp(reference, options), off)
                self.assertFalse(
                    set(off) & {'clock64', 'section_clocks', 'section_cycles', 'section_begin', 'barrier_begin'}
                )
                on = pp(source, options + ['-DEXP_SECTION_CLOCKS'])
                self.assertTrue({'clock64', 'section_clocks', 'section_cycles'} <= set(on))
                self.assertNotEqual(off, on)
        # Negative controls: off synchronization loss and leaked observer ABI must fail.
        options = ['-DFW=15', '-DK_FIRST=1', '-DK_PHASE=2', '-DK_BOUNDARY=1']
        for old, new in (
            ('#define SECTION_SYNC(op) op', '#define SECTION_SYNC(op)'),
            ('#define SECTION_PARAM\n', '#define SECTION_PARAM ,int*leaked_clock\n'),
        ):
            mutated = source.replace(old, new)
            self.assertNotEqual(pp(mutated, options), pp(without_clocks(mutated), options))

    def test_real_clock_macros_wrap_guard_and_actual_slot_expression(self):
        """Execute actual macros/index on host; synchronization remains a device limitation."""
        testkit.require_tool('clang++')
        source = paths.KERNEL_CU.read_text()
        start = source.index('#ifdef EXP_SECTION_CLOCKS\n#define SECTION_PARAM')
        macros = source[start : source.index('#endif', start) + len('#endif')]
        slot = re.search(r'int64_t slot=([^;]+);', source).group(1)
        program = (
            '''
#include <cstdio>
#include <cassert>
#include <initializer_list>
#define EXP_SECTION_CLOCKS
#define int64_t long long
struct Dim { int x, y; } blockIdx;
unsigned long long ticks[2]; int tick_index;
unsigned long long clock64() { return ticks[tick_index++]; }
'''
            + macros
            + '''
int main() {
  int lane = 0; unsigned long long section_begin = 0, section_cycles[8] = {};
  ticks[0] = (~0ULL) - 8; ticks[1] = 1; tick_index = 0;
  SECTION_BEGIN(); SECTION_END(0);
  assert(section_cycles[0] == 10 && tick_index == 2);
  tick_index = 0; SECTION_SYNC((void)0);
  assert(section_cycles[6] == 10 && tick_index == 2);
  lane = 1; tick_index = 0;
  SECTION_BEGIN(); SECTION_END(0); SECTION_SYNC((void)0);
  assert(tick_index == 0);
  lane = 0; SECTION_BEGIN_IF(false); SECTION_END_IF(1, false);
  assert(tick_index == 0 && section_cycles[1] == 0);
  int seen[5*15*8*8+2];
  for(int FW : {1, 15}) for(int TW : {1, 3, 5, 15}) {
    if(TW > FW || FW % TW) continue;
    for(int nwarps : {1, 2, 4, 8}) {
      int total=5*(FW/TW)*nwarps*8;
      for(int i=0;i<total+2;i++)seen[i]=-1;
      for(int c0 : {0, 2, 4}) for(int cell=0;cell<((c0==4)?1:2);cell++)
        for(int tile=0;tile<FW/TW;tile++) for(int warp=0;warp<nwarps;warp++) {
          blockIdx.y=tile;
          long long slot=EXPRESSION;
          long long expected=((c0+cell)*(FW/TW)*nwarps+tile*nwarps+warp)*8;
          for(int k=0;k<8;k++) {
            long long actual=c0*(FW/TW)*nwarps*8+slot+k;
            assert(actual == expected+k && actual>=0 && actual<total);
            assert(seen[actual+1] == -1); seen[actual+1]=k;
          }
        }
      assert(seen[0]==-1 && seen[total+1]==-1);
      for(int i=1;i<=total;i++)assert(seen[i]>=0);
    }
  }
  std::puts("actual macros and slot: PASS (host only)");
}
'''.replace('EXPRESSION', slot)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clock_contract.cpp'
            path.write_text(program)
            executable = Path(directory) / 'clock_contract'
            compiled = subprocess.run(
                ['clang++', '-std=c++17', str(path), '-o', str(executable)], capture_output=True, text=True
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            self.assertIn('PASS', subprocess.check_output([str(executable)], text=True))


class JudgeAdversarial(unittest.TestCase):
    """Exercise the unchanged public CLI using normative validation artifacts."""

    def validation(self):
        return json.loads((evidence() / 'prechange_selftests/A/validation.json').read_text())

    def judge(self, data):
        with tempfile.TemporaryDirectory() as directory:
            src, out = Path(directory) / 'validation.json', Path(directory) / 'judge.json'
            src.write_text(json.dumps(data))
            result = subprocess.run(
                [sys.executable, '-B', str(campaign_checks()), 'selftest-judge', str(src), str(out)],
                capture_output=True,
                text=True,
            )
            return result, json.loads(out.read_text()) if out.exists() else None

    def test_exact_metric_limits_pass_despite_informational_validation_status(self):
        data = self.validation()
        data['pass'] = False
        data['selftest']['max_relative'] = 1e-5
        for case in data['cuda_numpy_parity']['cases']:
            case['moment_l2_relative'] = 1e-6
        result, report = self.judge(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report['pass_'])
        self.assertFalse(report['validate_py_pass'])

    def test_either_metric_above_limit_nonfinite_or_missing_is_rejected(self):
        for metric, limit in (('physics', 1e-5), ('moment', 1e-6)):
            for value in (math.nextafter(limit, math.inf), math.nan, math.inf, None, 'PASS'):
                with self.subTest(metric=metric, value=value):
                    data = self.validation()
                    if metric == 'physics':
                        data['selftest']['max_relative'] = value
                    else:
                        data['cuda_numpy_parity']['cases'][0]['moment_l2_relative'] = value
                    result, report = self.judge(data)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(report['pass_'])
                    # The fixed reference requires nonzero exit and a false gate.
                    # Existing judge diagnostic max raises TypeError for strings
                    # after writing that false verdict (reported separately).
                    if not isinstance(value, str):
                        self.assertIn('E_SELFTEST_JUDGE', result.stderr)

    def test_missing_cases_identity_or_current_artifact_cannot_pass(self):
        for mode in ('empty', 'short', 'identity'):
            with self.subTest(mode=mode):
                data = self.validation()
                if mode == 'empty':
                    data = {}
                elif mode == 'short':
                    data['cuda_numpy_parity']['cases'].pop()
                else:
                    data['selftest']['identity_match'] = False
                result, report = self.judge(data)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(report['pass_'])
        for mode in ('missing', 'truncated'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                src, out = Path(directory) / 'validation.json', Path(directory) / 'judge.json'
                if mode == 'truncated':
                    src.write_text('{"selftest":')
                result = subprocess.run(
                    [sys.executable, '-B', str(campaign_checks()), 'selftest-judge', str(src), str(out)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
