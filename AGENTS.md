# Where to start

Paths are relative to `src/sosfuse2/` unless they start with `docs/`. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
describes the package structure and rules, and [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) lists the environment
variables.

| Task | Open first | Section |
|---|---|---|
| State and grid settings | `jobs/run_state.py::settings_for`, `jobs/run_state.py::resolve_settings`, `atmosphere/components.py::canonical_state` | [1](docs/ARCHITECTURE.md#1-one-input-state-and-one-output-table) |
| Transport and iteration | `solve/source_solver.py::Solver.solve`, `solve/source_solver.py::Kernel`, `backends/numpy_reference.py::transport` | [4](docs/ARCHITECTURE.md#4-the-call-spine) |
| Endpoint preparation and parallel workers | `solve/ray_geometry.py::PackedCache`, `solve/ray_geometry_parallel.py::workers`, `precompute/column_parallel.py::workers`, `precompute/host_cache.py` | [5](docs/ARCHITECTURE.md#5-topics-that-cross-layers) |
| Cache and code identity | `foundation/content_identity.py::package_hashes`, `foundation/paths.py::PRODUCTION_MODULES`, `precompute/endpoint_contract.py::identity` | [8](docs/ARCHITECTURE.md#8-moving-or-adding-a-file) |
| Observation table and previews | `outputs/radiance_table.py::make_table`, `outputs/pack_state.py::render_functions`, `outputs/preview_functions.py` | [4](docs/ARCHITECTURE.md#4-the-call-spine) |
| Kernel formulas | `backends/kernel.cu`, `backends/numpy_reference.py::extinction_profile`, `backends/sh_layout.py`, `docs/MODEL.md` | [4](docs/ARCHITECTURE.md#4-the-call-spine) |
| Stratosphere | `atmosphere/stratosphere.py`, `solve/strato_sh_aux.py`, `checks/stratosphere_checks.py` | [5](docs/ARCHITECTURE.md#5-topics-that-cross-layers) |
| New command | `__main__.py`, the `main` of the module it runs | [7](docs/ARCHITECTURE.md#7-commands-and-settings) |
| Gate baselines and anchors (changes need the maintainer's approval) | `checks/gate.py::pinned_hashes`, `checks/gate.py::code_identity`, `checks/gate_data/fixtures.json`, `docs/CHECKS.md` | [8](docs/ARCHITECTURE.md#8-moving-or-adding-a-file) |
| Move or add a file | `foundation/paths.py::PRODUCTION_MODULES`, `solve/engine.py::SOURCE_FILES`, `precompute/endpoint_contract.py::UPSTREAM` | [8](docs/ARCHITECTURE.md#8-moving-or-adding-a-file) |
| Layer rules and import exceptions | `tests/test_layers.py`, `tests/test_names.py` | [6](docs/ARCHITECTURE.md#6-known-upward-imports) |
| Regenerate the walkthrough | `jobs/walkthrough.py`, `tests/test_walkthrough.py`, `docs/WALKTHROUGH.md` | [3](docs/ARCHITECTURE.md#3-the-walkthrough) |
