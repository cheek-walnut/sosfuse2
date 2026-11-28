"""Ordered package inputs to source identities and cache keys.

Register each package-relative path once in FILES. Every Python row contributes to package
provenance and (unless gate_exclude is set) the gate code identity. An empty row is sufficient
for a module used only by those two consumers. Other fields give a zero-based position in a
consumer's input list. A tuple of positions hashes the file once per slot. shared_sun=True
selects a path for the lexically ordered shared table. column_name overrides a numeric
slot label in the column-cache source dictionary.

An engine cell includes engine_source, engine_extra and cell_extra. Both preview_functions
and pack_state therefore affect the cell key. The common endpoint includes endpoint_upstream,
the contract module itself, and the column method identity. Consumers retain their own hash
encoding and add their runtime inputs, such as grids, profiles and arrays.
"""

from functools import cache

CUDA_SOURCE = 'backends/kernel.cu'
NATIVE_SOURCE = 'backends/native/sh_cpu.cpp'

FILES = {
    '__init__.py': {},
    '__main__.py': {},
    'foundation/__init__.py': dict(host_module=0),
    'atmosphere/__init__.py': {},
    'rays/__init__.py': {},
    'precompute/__init__.py': {},
    'backends/__init__.py': {},
    'solve/__init__.py': dict(accepted=0),
    'outputs/__init__.py': {},
    'jobs/__init__.py': {},
    'checks/__init__.py': {},
    'asset_tools/__init__.py': {},
    'foundation/available_cpus.py': dict(host_module=1),
    'foundation/top.py': dict(
        engine_source=0,
        endpoint_upstream=0,
        column_integral=2,
        shared_sun=True,
        sun_table=1,
        worker=0,
        strato_geometry=0,
    ),
    'atmosphere/upper_air.py': {},
    'asset_tools/generate_upper_air.py': {},
    'backends/flags_env.py': {},
    'solve/engine.py': dict(
        accepted=1,
        engine_extra=0,
        endpoint_upstream=1,
        shared_sun=True,
        column_name='engine',
        density_columns=5,
        strato_columns=5,
    ),
    'outputs/pack_state.py': dict(host_module=2, engine_extra=1),
    'atmosphere/air_wavelength.py': {},
    'backends/arithmetic.py': {},
    'jobs/cache_store.py': {},
    'precompute/column_cache.py': {},
    'precompute/column_parallel.py': dict(
        endpoint_upstream=10,
        shared_sun=True,
        sun_table=7,
        host_context=7,
        worker=9,
        strato_geometry=6,
    ),
    'rays/column_table.py': dict(column_integral=0, sun_table=5, host_context=6, strato_geometry=5),
    'backends/compile_cache.py': {},
    'atmosphere/components.py': dict(cell_extra=1, endpoint_upstream=8),
    'atmosphere/density.py': dict(
        cell_extra=0,
        endpoint_upstream=7,
        shared_sun=True,
        sun_table=2,
        host_context=1,
        worker=7,
        strato_geometry=4,
        density_columns=0,
    ),
    'backends/device_profile.py': {},
    'backends/vram_budget.py': {},
    'backends/vram_trace.py': {},
    'solve/source_solver.py': {},
    'solve/source_iteration.py': {},
    'checks/gate.py': dict(gate_exclude=0),
    'checks/kernel_diagnostics.py': {},
    'asset_tools/generate_density_profile.py': {},
    'asset_tools/generate_minor_absorbers.py': {},
    'asset_tools/generate_strato_optics.py': {},
    'asset_tools/generate_tropo_optics.py': {},
    'solve/ray_geometry.py': dict(query=1),
    'precompute/endpoint_contract.py': dict(endpoint_contract=0),
    'solve/ray_geometry_cpu.py': {},
    'solve/ray_geometry_parallel.py': dict(endpoint_upstream=9, worker=6),
    'precompute/ground_parallel.py': dict(host_context=5, host_aerosol=0, worker=11, strato_geometry=2),
    'solve/ground_sh.py': {},
    'precompute/host_cache.py': dict(host_context=0),
    'foundation/host_limits.py': {},
    'foundation/content_identity.py': {},
    'asset_tools/iprt.py': {},
    'backends/flags.py': {},
    'solve/ksplit.py': {},
    'asset_tools/mie.py': {},
    'atmosphere/minor_absorbers.py': {},
    'checks/numerical_checks.py': {},
    'atmosphere/ozone_temperature.py': {},
    'foundation/paths.py': {},
    'asset_tools/prepare_minor_absorbers.py': {},
    'rays/shell_segments.py': dict(
        engine_source=1,
        endpoint_upstream=2,
        shared_sun=True,
        sun_table=0,
        worker=1,
        strato_geometry=8,
        density_columns=(1, 2, 3),
        strato_columns=(1, 2, 3),
    ),
    'atmosphere/extinction.py': dict(engine_source=2),
    'rays/incoming_quadrature.py': dict(engine_source=3, host_context=3, worker=2, strato_geometry=7),
    'rays/incoming_rings.py': dict(engine_source=4, endpoint_upstream=3, worker=3),
    'rays/zenith_interpolation.py': dict(engine_source=5),
    'solve/engine_core.py': dict(engine_source=6, endpoint_upstream=4, worker=5),
    'outputs/preview_functions.py': dict(engine_source=7),
    'jobs/pipeline.py': {},
    'jobs/pipeline_execution.py': dict(query=2),
    'jobs/plan_queue.py': {},
    'jobs/precompile.py': {},
    'checks/validation_evidence.py': {},
    'rays/ray_columns.py': dict(
        endpoint_upstream=6,
        shared_sun=True,
        sun_table=4,
        host_context=2,
        worker=4,
        strato_geometry=3,
        column_name='joint_columns',
        density_columns=4,
        strato_columns=4,
    ),
    'atmosphere/rayleigh.py': {},
    'backends/numpy_reference.py': {},
    'checks/solver_diagnostics.py': {},
    'jobs/run_state.py': dict(endpoint_upstream=11),
    'solve/configuration.py': dict(endpoint_upstream=12),
    'solve/build.py': dict(endpoint_upstream=13),
    'jobs/run_storage.py': {},
    'jobs/walkthrough.py': {},
    'backends/sh_layout.py': {},
    'atmosphere/spectral.py': {},
    'solve/strato_sh_aux.py': {},
    'asset_tools/strato_states.py': {},
    'backends/mlx_backend.py': {},
    'backends/mlx_ops.py': {},
    'backends/mlx_query.py': {},
    'backends/mlx_query_kernel.py': {},
    'backends/mlx_transport.py': {},
    'backends/mlx_warm.py': {},
    'backends/mlx_warm_kernels.py': {},
    'atmosphere/stratosphere.py': dict(worker=8, strato_geometry=1, strato_columns=0),
    'precompute/sun_parallel.py': dict(shared_sun=True, sun_table=6, host_context=4, worker=10),
    'outputs/radiance_table.py': dict(query=0),
    'outputs/quantized_table.py': {},
    'jobs/tables_from_sources.py': {},
    'foundation/timers.py': {},
    'foundation/version.py': {},
    'atmosphere/tropo_mie.py': {},
    'atmosphere/tropo_profile.py': dict(
        engine_source=8,
        endpoint_upstream=5,
        column_integral=1,
        shared_sun=True,
        sun_table=3,
        host_aerosol=1,
    ),
    'checks/validate.py': {},
    'checks/stratosphere_checks.py': {},
    'solve/ksplit_tolerance.py': {},
    'foundation/dependencies.py': {},
    CUDA_SOURCE: {},
    NATIVE_SOURCE: dict(engine_extra=2),
    'assets/spectra.json': dict(asset=0, engine_source=9),
    'assets/spectra_fine.json': dict(asset=1),
    'assets/spectra_fine_temperature.json': dict(asset=2),
    'assets/lef_cie.json': dict(asset=3),
    'assets/cmf_1nm.json': dict(asset=4),
    'assets/states_v1.json': dict(asset=5, state_asset=0),
    'assets/states_v2.json': dict(asset=6, state_asset=1),
    'assets/states_v1_usstd.json': dict(asset=7, state_asset=2),
    'assets/rgb_weights.npy': dict(asset=8, engine_extra=3),
    'assets/temperature_usstd76.json': dict(asset=9),
    'assets/no2_afglus_profile.json': dict(asset=10, shared_sun=True),
    'assets/no2_vandaele1998_fine.json': dict(asset=11),
    'assets/upper_air_us76_neff.json': dict(asset=12),
    'checks/gate_data/fixtures.json': dict(gate_fixture=0),
    'backends/configs/C3.json': dict(gate_config=0),
    'backends/configs/C3_nostrato.json': dict(gate_config=1),
    'backends/configs/devices.json': dict(gate_config=2),
}


@cache
def files(scope):
    """Return paths in this consumer's order, preserving repeated inputs."""
    if scope == 'shared_sun':
        return tuple(sorted(path for path, scopes in FILES.items() if scopes.get(scope)))
    ordered = {}
    for path, scopes in FILES.items():
        if scope not in scopes:
            continue
        positions = scopes[scope]
        if isinstance(positions, int):
            positions = (positions,)
        for index in positions:
            if index in ordered:
                raise ValueError(f'Duplicate {scope} position {index}: {ordered[index]} and {path}')
            ordered[index] = path
    if not ordered:
        raise KeyError(scope)
    return tuple(ordered[index] for index in sorted(ordered))


def column_slots(density):
    """Numeric string slots and named slots in the column-cache source dictionary."""
    names = files('density_columns' if density else 'strato_columns')
    for index, name in enumerate(names):
        yield FILES[name].get('column_name', str(index)), name
