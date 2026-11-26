# Physical model

sosfuse2 computes the radiance of the clear sky seen by an observer near the ground, for a Sun that may stand above the
horizon or far below it. The atmosphere is a spherical shell with horizontally uniform optical properties, the Sun is a
point source, and multiple scattering is summed to convergence, with polarised Rayleigh scattering and depolarising
aerosol scattering. The output is total radiance. `REFERENCES.md` lists the literature and `THIRD_PARTY.md` the data
files.

## The radiance table

A run writes one table per atmospheric state, `table.npz`. Its array `radiance` is float32 in W m⁻² sr⁻¹ nm⁻¹ with the
axes (solar elevation, view elevation, relative azimuth, wavelength). The default axes are:

| axis | values |
|---|---|
| solar elevation | +6° to −18° in 1° steps; −3° and −6° are required rows |
| view elevation | 100 rows of 0.9°, centres 89.55° to 0.45°; the horizon is the lower edge of the last row |
| relative azimuth | 180 columns of 1°, centres 0.5° to 179.5°; 0° is towards the Sun, 180° away from it |
| wavelength | 45 bands between 360 and 830 nm, centres stored as `wavelength_nm` |

Because the field is symmetric about the solar plane, only one half of the sky is stored. Each value is a point evaluation
at a row and column centre rather than an average over the pixel. The observer stands 0.1 km above the ground, and solar
elevation is measured from the horizontal plane at the observer, negative below the horizon.

The table holds total radiance. Polarisation acts inside the calculation, as described under *Rayleigh scattering and
polarisation*, but the code writes no Stokes parameters, degree of polarisation or polarisation angle.

## Geometry

The Earth is a sphere of radius 6371 km with a Lambertian surface. The default atmosphere ends at 80 km. Rays are
straight lines, so refraction is not modelled. The Sun is a point at infinity without a disc or a penumbra,
and a point is lit when the straight line from it to the Sun misses the Earth, which makes the Earth's shadow a sharp
cylinder.

Because the atmosphere is uniform in the horizontal, the radiation field is symmetric about the direction to the Sun, and
the solver stores it as a function of height `h` and of the angle `χ` from the sub-solar point. An observer at solar
elevation `h_s` sits at `χ = 90° − h_s`, so changing the solar elevation of a table row moves the observer along the `χ`
axis without changing the solved field. The `χ` grid reaches 180°, which leaves the geometry with no limit on solar
depression; the convergence test of the iteration looks only at +2°, −3°, −6°, −9° and −12° (see *Iteration*), and the
default table stops at −18°.

### Calculation boundary

The top-level `top_km` key in `--settings-json` can extend the full grid above 80 km up to 200 km. The top must be a
multiple of the profile spacing. For an extended top, each state in `--states` needs an explicit `density` block with `density_profile` set to `usstd76`. For example, use
`{"density":{"density_profile":"usstd76","ozone_profile":"usstd76","rayleigh_column":"usstd"}}`. External profiles
are unsupported with an extended top. When `top_km` is a multiple of 10 km, the default upper height nodes
are spaced every 10 km from 90 km to the top. Otherwise supply `upper_heights_km`. A supplied list must increase
strictly above 80 km and end at `top_km`.

For example, a state with `density.profile_step_km` set to `0.125` can run with
`{"top_km":80.125,"upper_heights_km":[80.125]}` in `settings.json`. To precompile that grid, use
`python -m sosfuse2 precompile --cache cache --settings-json settings.json --profile-step-km 0.125`.
The precompiler reads the profile spacing from the command line and defaults to 0.5 km.

Above 80 km, the default `upper_air: us76_neff` extends Rayleigh scattering with a tabulated effective air density
matched to the coefficient at 80 km. `upper_air: vacuum` leaves the extension empty. The other constituents end at
or below 80 km. The scattering source of the stratospheric layer is a value per height node that conserves the layer's
column over the cells of the grid, so the node at 80 km changes with the cell above it: an empty extension reproduces
the 80 km result only for a layer with no mass in the last cell below 80 km. The bundled layer (21 km, σ 3 km) has
none and its node values are equal with and without the extension; a layer centred near 77 km is not equivalent. The extended grid adds the top shell to the incident-direction quadrature, adds −15° to the stopping
and acceleration samples, and records −18° in the convergence history.

## Constituents

Six constituents extinguish or scatter light: air molecules, a tropospheric aerosol, ozone, NO₂, O₄ and, if enabled, a
stratospheric sulfate layer. Each extinction coefficient is the product of a vertical profile and a wavelength law, with
the square of a profile in the case of O₄. NO₂ and O₄ arrived in 1.1.0 together with a new Rayleigh law. A state file
sets the parameters; `src/sosfuse2/assets/states_v2.json` holds 64 states sampled over the axes given in the tables below.

### Molecules

Molecular scattering follows

$$
\beta_R(h,\lambda) = \beta_{450}\,\rho_R(h)\,\frac{\sigma_B(\lambda_{\mathrm{vac}})}{\sigma_B(\lambda_{\mathrm{vac}}(450\ \mathrm{nm}_{\mathrm{air}}))}
$$

where `σ_B` is the Rayleigh cross section of Bodhaine et al. (1999), their equations 18, 19, 5, 6, 23 and 22 with 360 ppm
CO₂, evaluated at the vacuum wavelength of the engine's standard-air wavelength λ (Ciddor 1996). Because it includes the
dispersion of the refractive index and the King factor of air, it is not a pure power law (`src/sosfuse2/atmosphere/rayleigh.py`).
`β₄₅₀` is the 1.0.0 coefficient at 450 nm, which 1.1.0 keeps. Version 1.0.0 used `β₅₅₀ (550 nm / λ)⁴`, and against that λ⁻⁴
law the new coefficient is 1.037 times larger at 360 nm, equal at 450 nm, 0.973 times at 600 nm and 0.958 times at
830 nm. The depolarisation factor of the phase function stays at 0.03 (see *Rayleigh scattering and polarisation*). The
vertical profile `ρ_R` depends on the density option:

| option | profile | Rayleigh optical depth, 0 to 80 km, at 450 nm / 550 nm |
|---|---|---|
| `exp8`/`triangle`/`keep` (default through 1.4.0) | exp(−h / 8 km) | 0.2420 / 0.1063 (β₄₅₀ = 0.030255 km⁻¹) |
| `usstd76`/`usstd76`/`usstd` (default) | hydrostatic number density of the US Standard Atmosphere 1976 (NOAA et al. 1976) | 0.2215 / 0.0972 |

The reference fixtures and `states_v1_usstd.json` use the second row. Its optical depth at 450 nm is the sum over the layers
below 80 km of the tabulated Rayleigh optical depths of the IPRT phase E atmosphere at 450 nm (IPRT project: Emde et al.
2015), so the Rayleigh spectrum is the Bodhaine law anchored at that 450 nm value. With the same anchors, the λ⁻⁴ law of
1.0.0 gave 0.1085 and 0.0993 at 550 nm for the two rows. The states in `states_v1.json` and `states_v2.json` carry no
density block and take the second row. Below −6° solar elevation the exp8 configuration is 1.4 to 2.1 times too bright
against the Maryland and Paranal measurements. The default still has site-dependent errors; README *Limitations*
and `docs/VALIDATION.md` Table 13 give the comparison. A state's density block and the density command-line options
together must supply all three choices.

### Ozone

Ozone is a pure absorber with extinction coefficient

$$
\beta_{\mathrm{O_3}}(h,\lambda) = \bar\sigma(\lambda)\,\frac{N}{15\ \mathrm{km}}\,q(h),\qquad \int_0^{80\,\mathrm{km}} q\,\mathrm{d}h = 15\ \mathrm{km}
$$

where the column `N` is `ozone_du` × 2.687×10²⁰ m⁻² per Dobson unit and the state axis `ozone_du` runs from 220 to 420.
The shape `q` has two options. The `triangle` option is `max(0, 1 − |h − 25 km| / 15 km)`, which is non-zero between
10 and 40 km and peaks at 25 km. The default `usstd76` option takes the shape from the layer absorption at 320 nm of the same IPRT
atmosphere; the density asset records that this is a vertical shape normalised to the state's column and not a measured
ozone profile.

The cross section `σ̄` comes from the Serdyuchenko and Gorshelev laboratory data (Serdyuchenko et al. 2014, Gorshelev et
al. 2014). In the default band mode of the full grid, it is first averaged over each band and then over height with the
weight `q(h)`, using the temperature of the US Standard Atmosphere at each height and linear interpolation in temperature
between the columns from 193 K to 293 K. This gives one number per band, used at all heights. A fixed 233 K column can be
selected instead. The reduced smoke grid at 550 nm uses the older 48-wavelength table of `assets/spectra.json`, Bruneton's
demo data (`THIRD_PARTY.md` §3).

### NO₂

NO₂ is also a pure absorber, with extinction coefficient

$$
\beta_{\mathrm{NO_2}}(h,\lambda) = \bar\sigma_{\mathrm{NO_2}}(\lambda)\,\frac{N_{\mathrm{NO_2}}}{H_f}\,f(h),\qquad H_f = \int_0^{80\,\mathrm{km}} f\,\mathrm{d}h
$$

where the column `N_NO2` is `no2_du` × 2.687×10²⁰ m⁻². The state parameter `parameters.no2_du` defaults to 0.207 DU, the
column of the reference profile, and 0 switches NO₂ off. The shape `f` is the NO₂ number density of the AFGL US Standard
profile (`afglus.dat`, Anderson et al. 1986, AFGL-TR-86-0110, as distributed with libRadtran 2.0.6), interpolated
log-linearly onto 0.5 km steps and normalised to a maximum of 1 (`assets/no2_afglus_profile.json`). It applies with every
density option, including an external air and ozone profile. The cross section is that of Vandaele et al. (1998) at
220 K, from BIRA-IASB data resampled to 0.1 nm cells (`assets/no2_vandaele1998_fine.json`; attribution and changes in
`THIRD_PARTY.md` §11). In band mode `σ̄` is the mean over each band and in point mode the value at the wavelength; it
applies at all heights because the cross section has no temperature dependence.

### O₄

The collision-induced absorption of O₂–O₂ pairs (O₄) is proportional to the square of the O₂ number density:

$$
\beta_{\mathrm{O_4}}(h,\lambda) = \bar\sigma_{\mathrm{O_4}}(\lambda)\,\bigl(0.20946\,n_{\mathrm{air}}(h)\bigr)^2,\qquad n_{\mathrm{air}}(h) = n_0\,\frac{\rho_R(h)}{\rho_R(0)}
$$

`n₀` is the air number density at the surface. It comes from the density asset for `usstd76`. For
the exp(−h / 8 km) profile (default through 1.4.0) it equals 2.5469×10²⁵ m⁻³ and the O₄ shape is then exp(−h / 4 km).
An external profile declares it as `air_surface_number_density_m3`. The state parameter `parameters.o4`, a JSON boolean that defaults to true,
switches O₄ on and off. The cross section is the HITRAN 2024 collision-induced absorption (Gordon et al. 2026; Terragni
et al. 2025): data set 54 at 293 K up to 499.861 nm, then data set 32 at 293 K up to 600 nm and at 287 K up to 653.842 nm,
resampled to 0.1 nm cells and zero outside that range. The O₂ A- and B-band sets are excluded. Like NO₂, `σ̄` is a band or
point value with no temperature dependence. The O₄ data are not bundled; `python -m sosfuse2.asset_tools.prepare_minor_absorbers`
prepares them once in the work root (README, *Quick start*; sources and changes in `THIRD_PARTY.md` §12).

### Tropospheric aerosol

Five state parameters describe the tropospheric aerosol.

| parameter | meaning | axis in `states_v2.json` |
|---|---|---|
| `aod550` | optical depth at 550 nm, 0 to 80 km | 0.02 to 0.6, logarithmic |
| `angstrom` | exponent α of the extinction law (550 nm / λ)^α | 0.8 to 2.0 |
| `ssa` | single-scattering albedo, the same at all wavelengths | 0.85 to 0.98 |
| `scale_height_km` | `H` of the profile exp(−h / H) | 1 to 3 |
| `g` | asymmetry parameter of the Henyey–Greenstein phase function, the same at all wavelengths | 0.65 to 0.8 |

The extinction coefficient is `aod550 · exp(−h/H) / ∫₀⁸⁰ exp(−h/H) dh · (550 nm / λ)^α`, and the scattering coefficient
is `ssa` times that. The phase function is that of Henyey and Greenstein (1941),
`p(cos Θ) = (1 − g²) / [4π (1 + g² − 2g cos Θ)^{3/2}]`.

Two optional state blocks replace parts of this default. `troposphere_phase` swaps the Henyey–Greenstein function for a
tabulated Mie phase function and leaves extinction, albedo and Ångström law as the state sets them. The shipped table is a
bimodal lognormal rural aerosol at 70 % relative humidity, built from the Shettle and Fenn (1979) rural model with the
mode parameters checked against LOWTRAN 7 tables (Kneizys et al. 1988), computed with Bohren–Huffman spheres (Bohren and
Huffman 1983) at 18 wavelengths between 360 and 830 nm and interpolated linearly in wavelength. `troposphere_profile`
replaces the single exponential by a boundary layer and a free troposphere joined by a smooth step, centred at 2 km by
default. No shipped state file carries either block.

### Stratospheric sulfate layer

The layer is enabled by default in all 64 bundled `states_v2.json` states. `states_v1.json` keeps the stratosphere status `unsupported` and
`states_v1_usstd.json` keeps `disabled`; the runner can enable either with `--stratosphere on`. Test fixtures enable
the layer in some states. When on, its profile is a Gaussian, by default centred at 21 km with σ = 3 km and normalised over 0 to 80 km,
and its optical depth at 550 nm is `stratosphere_tau550`, with an axis of 0.002 to 0.006 in `states_v2.json`. The
particles are a single lognormal sulfate population with median radius 0.08 µm, geometric width 1.6 and refractive index
1.43 + 0i, so they do not absorb. The shipped Mie table covers 360 to 830 nm at 18 wavelengths, and extinction at other
wavelengths scales with the Mie cross section relative to 550 nm. The asymmetry parameter of the table falls from 0.713 at
360 nm to 0.510 at 830 nm. The layer scatters with its own tabulated phase function in place of Henyey–Greenstein.

### Ground

The surface is Lambertian with an albedo `albedo`, axis 0.05 to 0.8, that does not depend on wavelength. It reflects the
direct solar beam and the diffuse radiation reaching it, and the atmosphere scatters the reflected light again. There is
no topography and no directional reflectance.

## Scattering

### Rayleigh scattering and polarisation

For intensity, Rayleigh scattering uses the phase function

$$
p(\Theta) = \Delta\,\frac{3}{16\pi}\left(1 + \cos^2\Theta\right) + \frac{1-\Delta}{4\pi},\qquad \Delta = \frac{1-\delta}{1+\delta/2}
$$

with a constant depolarisation factor δ = 0.03, so that Δ = 0.9557. The solver carries the radiance field as a symmetric
3×3 coherency tensor whose trace is the intensity. Reflection symmetry about the solar plane leaves four independent
components, stored in a local frame at each cell. Molecular scattering acts on the full tensor as dipole scattering with
the depolarisation factor δ, so polarisation changes the intensity at every order. The runner option
`--polarization-model` accepts `preserving`, but only the `depolarizing` model is implemented, and the other value stops
the run with `E_UNSUPPORTED_POL_MODEL`.

### Aerosol scattering

Tropospheric aerosol and stratospheric sulfate scatter with scalar phase functions and depolarise completely: the
scattered light is unpolarised, with its intensity given by the scalar phase function, whatever the polarisation of the
incident light. The tabulated aerosol phase functions are the first element of the Mie scattering matrix, and the other
elements play no part in the transport.

### Truncation of phase functions

The direct beam is scattered once with the full phase function. For tropospheric aerosol that is the Henyey–Greenstein
function or the tabulated Mie function, and for the stratosphere it is the Legendre series of the Mie phase function to
degree 32. In the multiple-scattering source, the phase function of each aerosol is written as
`p(cos Θ) = (1/4π) Σ χ_l P_l(cos Θ)` and cut at degree `L`, which is 16 in the production setting and may be set to 24 or
32. The coefficients are `χ_l = (2l+1) g^l` for Henyey–Greenstein and the tabulated moments for Mie. Because there is no
delta-M or other renormalisation of the forward peak, the truncated terms are simply dropped. Rayleigh scattering needs no
expansion: its phase function is a quadratic polynomial in direction, and the four-component tensor holds it exactly.

## Solving the transport

### Single scattering

Sunlight scattered once into the line of sight is integrated along that line. The optical depth from the Sun to the
scattering point comes from a precomputed table of slant columns of each constituent along the solar direction, as a
function of height and angle, and the optical depth along the line of sight is integrated with Simpson's rule between the
quadrature nodes. Lines of sight are cut where they cross the 10, 25 and 40 km spheres, the Earth's limb, the shadow
boundary and the ground, and each piece carries 12 Gauss–Legendre nodes, or 24 for the lines of sight of the output
table. Light reflected once by the ground belongs to the first order and is scattered with the untruncated phase
functions. Setting `"export_first_order": true` in `--settings-json` saves `first_order.npz` as a separate archive
member with the same axes as the total radiance table, including the ground reflection of direct light.
For a custom `radiance_table --out sky.npz`, the companion files are `sky.npz.first_order.npz` and
`sky.npz.meta.json`. The default output name `table.npz` uses `first_order.npz` and `meta.json`.

### Multiple scattering

Higher orders are carried by a source field defined on the (height, `χ`) grid. In each cell the source is stored as ten
Rayleigh slots, four tensor components and six spherical-harmonic coefficients, followed by the expansion coefficients
of the aerosol scattering source, with a second set for the stratosphere if it is enabled. One application of the
transport operator computes, for each cell, the radiance
arriving from every incident direction by integrating the source field along that direction, with attenuation by all
constituents and with the light reflected from the ground where the ray ends on it. The source is interpolated linearly
in height and `χ` and then rescaled by `(G/A)^W`, where `G` and `A` are the weighted geometric and arithmetic means of the
corner traces that are at least `FLT_MIN` and `W` is their total weight. With all corners valid the trace is the
geometric mean, and as `W` goes to 0 the result goes continuously to the linear interpolation. The arriving radiance is
then projected onto the source basis, which gives the next source field. The incident directions are Gauss–Legendre nodes
inside intervals whose ends are the limb, the local horizon, the shadow boundary and the directions tangent to the shells
at 0, 1.2, 8, 25, 40 and 80 km on the default grid.

### Iteration

The source field is the fixed point of this operator. One application adds one order of scattering, so plain iteration
from zero is the successive-orders method. The production runner instead accelerates the iteration with Anderson mixing
(Anderson 1965) of depth 4, applied to each wavelength separately. A wavelength has converged when, from the second
evaluation onward, both the change of the source field and the change of the observer radiance between successive
iterates are at most `anderson_tol` in relative terms. This setting defaults to 1e-3 and accepts numbers from 1e-6 to
1e-2 inclusive. The observer test uses 35 radiances: seven viewing directions for each of
the five solar elevations +2°, −3°, −6°, −9° and −12°, each averaged over a 2° × 2° box. The runner stops with
`E_NOT_CONVERGED` if a wavelength has not converged within the evaluation limit, 64 by default. Because Anderson mixing
combines earlier iterates, the iterates are not scattering orders, and the number recorded in `meta.json` counts
evaluations.

Once the source field has converged, the observer radiance for every row of the table is computed by integrating the
converged source and the single-scattering term along the lines of sight. All solar elevations share the same source
field.

### Reference implementation and solver modules

A NumPy reference implements the same transport with double-single arithmetic for the ray geometry and FP64 projection
sums, and serves as the golden reference of the numeric checks. The CUDA transport uses FP32 ray geometry, spherical
harmonics and compensated projection sums, with the ray and segment constants prepared in FP64 on the host. The
README and `docs/CHECKS.md` describe the tolerances and checks. The solver definitions in `src/sosfuse2/rays/`, `src/sosfuse2/atmosphere/extinction.py` and
`src/sosfuse2/solve/engine_core.py` define the grids, the quadrature and the physical constants, while
`src/sosfuse2/atmosphere/rayleigh.py` and `src/sosfuse2/atmosphere/minor_absorbers.py` hold the Rayleigh and
minor-absorber coefficients of 1.1.0.

## Production grid

| quantity | value |
|---|---|
| heights | 44 nodes from 0 to 80 km: 20 base nodes (0, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2, 3, 4.5, 6.5, 9, 13, 18, 26, 40, 60, 80 km), their midpoints below 13 km, and two interior thirds of each interval above 13 km |
| `χ` | 75 nodes: 10° steps to 80°, 1° to 85°, 0.5° from 85° to 110°, 2° to 118°, then 10° steps from 120° to 180° |
| incident directions per cell | 12 Gauss–Legendre nodes per polar interval, 16 azimuth intervals over 0° to 180° |
| line-of-sight nodes | 12 per piece for the source, 24 per piece for the output |
| Legendre degree `L` | 16 |
| solar-column table | resolution scale 4 |
| Anderson relative tolerance (`anderson_tol`), depth | 1e-3 by default, 4 |

The table lists the default grid, and the settings `h_refine`, `nr`, `chi_step_deg`, `chi_refine_deg` and `nchi` change
its height and `χ` axes. Extended grids append the upper height nodes to the refined grid. In the settings,
`h_refine` takes `b<m>a<k>`, which splits each height interval at or below 13
km into m parts and each interval above into k parts, or `b<m>a<k1>.<k2>.<k3>.<k4>.<k5>`, which gives the five intervals
above 13 km their own counts, with m and k from 1 to 12. The names `none`, `mid` and `below13mid_above13thirds` are
aliases of `b1a1`, `b2a2` and `b2a3`; the default 44 nodes are `b2a3`. For the `χ` grid, `chi_step_deg` takes 0.5 by
default, 0.25 or 0.125. The optional `chi_refine_deg` takes one `[lo, hi]` pair or a nonempty list of pairs. In
list order, each pair inserts the midpoint of every adjacent node pair in its range. Its endpoints must be distinct,
ordered nodes of the grid as refined so far, with 85 ≤ lo < hi ≤ 110. The counts `nr` and `nchi` are filled from the rules when omitted and must match them when given. The
reduced grid stays fixed. For example, `b2a6` gives 59 nodes, and `chi_refine_deg` `[88, 100]` on the 0.5° grid gives 99
`χ` nodes.

The runner's reduced option uses 4 heights, 10 angles and one wavelength (550 nm). It is a software check, and its grid is
too coarse for physical use.

### Finer and taller grids together

The production grid is the default. The settings below, saved as `settings.json`, raise all of its resolution options at
once, with a 200 km top, 59 height nodes up to 80 km, two refinement windows in `χ`, Legendre degree 24 and an Anderson
tolerance of 2e-4.

```json
{"top_km": 200, "h_refine": "b2a6", "chi_refine_deg": [[88, 104.5], [104, 104.25]], "L": 24, "anderson_tol": 0.0002}
```

The runner resolves it to 71 height nodes and 109 `χ` nodes, with `upper_air` left at `us76_neff`. The 71 heights are
the 59 nodes up to 80 km and 12 more every 10 km from 90 to 200 km. The first `χ` window inserts the midpoints between
88° and 104.5°, which halves the spacing there to 0.25°, and the second adds 104.125°. `meta.json` records the resolved
grid. As described under *Calculation boundary*, the extended top adds −15° to the stopping and acceleration samples and
records −18°.

Each state needs the explicit `density` block given under *Calculation boundary*. The states in `states_v2.json` carry
no such block, so the first command below writes a copy of the file with the block added to every state. The copy drops
the `parent` entry, which names `states_v1.json` by a path relative to the file, so it holds the 64 states of
`states_v2.json` and no others.

```sh
python - <<'EOF'
import json, pathlib, sosfuse2
doc = json.loads((pathlib.Path(sosfuse2.__file__).parent / 'assets' / 'states_v2.json').read_text())
doc.pop('parent')
for state in doc['states']:
    state['density'] = {'density_profile': 'usstd76', 'ozone_profile': 'usstd76', 'rayleigh_column': 'usstd'}
pathlib.Path('states.json').write_text(json.dumps(doc, indent=1))
EOF
SOSFUSE2_ANDERSON_HISTORY=host SOSFUSE2_GROUND_SH_L=24 python -B -m sosfuse2.jobs.run_state --id clear_80 \
  --states states.json --settings-json settings.json --backend cupy --work-root ./work
```

The two environment variables are separate from the settings file, and `docs/ENVIRONMENT.md` describes both.
`SOSFUSE2_ANDERSON_HISTORY=host` keeps the Anderson history in host memory, so the solve holds less device memory,
although the memory admission estimate does not change; an eligible CUDA solve copies one wavelength's history at a
time to the device for the products and the mixing.
`SOSFUSE2_GROUND_SH_L=24` sets the order of the ground convolution, which is off by default and changes the result only
for a state with nonzero albedo.

The runner stops with an error in each of these cases.

| case | error |
|---|---|
| `top_km` above 80 for a state without a `density` block, with a `density_profile` other than `usstd76` or with an external profile | `E_SETTINGS_TOP` |
| `top_km` above 80 with the reduced grid (`--reduced`) | `E_SETTINGS_TOP` |
| `nr` or `nchi` given with a value other than the counts above | `E_SETTINGS_GRID` |
| `SOSFUSE2_KSPLIT=2` or `3` with `L` above 16 | `E_KSPLIT_FLAGS` |
| `SOSFUSE2_GROUND_SH_L` set to a nonzero order for a state with an enabled `troposphere_phase` block or without an enabled stratospheric layer of nonzero optical depth | `E_GROUND_SH_PHASE` |
| a wavelength that has not converged within the evaluation limit | `E_NOT_CONVERGED` |

With `L` above 16 and `SOSFUSE2_KSPLIT` unset, the solver uses the unsplit kernel, and `meta.json` records
`ksplit_effective` 0.

Each key in the settings file can also be given alone. A `top_km` above 80 still needs the `density` block, and the
other keys need nothing else. The production queue, `python -B -m sosfuse2.jobs.pipeline`, takes the same
`--settings-json`, and its workers inherit the two environment variables.

To confirm that a run used what was requested, compare these entries of its `meta.json` with the settings file and
the environment.

| requested | entry in `meta.json` |
|---|---|
| `top_km`, the upper air and the height and `χ` grids | `settings.nr`, `settings.nchi` and `grid`, and for a top above 80 km `settings.top_km` and `settings.upper_air` |
| `L` | `settings.L` and `solve_grid.L` |
| `anderson_tol` | `settings.anderson_tol` and `anderson_tol` |
| `SOSFUSE2_ANDERSON_HISTORY=host` | `timings[i].anderson_history` for each chunk |
| `SOSFUSE2_GROUND_SH_L` | `settings.kernel.ground_sh.order`, and `timings[i].ground_sh_order` for each chunk, the order of the ground table the CUDA kernel used (0 without one, and with the NumPy backend, which convolves the ground exactly) |
| the density block | `density` |

If any of these differs, the run did not use the requested settings and its table should not be used as such. A
wavelength that does not converge stops the run with `E_NOT_CONVERGED`, so a run that writes its table has converged at
the recorded tolerance. Two changes of the kernel path leave these entries as requested and are recorded instead. One
is the unsplit kernel described above, with `ksplit.reason` `sh_order_gt_16` for `L` above 16 or
`state_minimum_exceeds_solo_budget` when the split records do not fit in device memory. The other is the splitting of a
transport evaluation into several launches under the device profile's time limit, whose profile and launch count are
in `kernel.launch`. `docs/ENVIRONMENT.md` lists the `ksplit` reasons and the device profiles.

## Spectral treatment and colour

The default wavelength grid has 45 bands between 360 and 830 nm. It starts from 40 bands of 11.75 nm, five of which
(489.25 to 501, 512.75 to 524.5, 524.5 to 536.25, 559.75 to 571.5 and 642 to 653.75 nm) are each split into two halves.
The solver runs at the centre wavelength of each band. The solar spectral irradiance and the ozone, NO₂ and O₄ cross
sections are band means taken from a 0.1 nm grid, whereas the Rayleigh and aerosol coefficients and the Mie tables are
evaluated at the band centre without averaging over the band.

The solar spectrum is the TSIS-1 Hybrid Solar Reference Spectrum version 2 (Coddington et al. 2021, 2023), whose file
summary states a scaling of all wavelengths by 0.999061 to the total solar irradiance measured by TSIS-1 (recorded in
`spectra_fine.json`). We converted it from vacuum to standard-air wavelengths with the equations of Ciddor (1996),
scaling the irradiance per wavelength by dλ_vac/dλ_air, and averaged it to 0.1 nm cells on the air axis. The
ozone data carry the same wavelength conversion. No correction for the distance between the Earth and the Sun is applied. The older
legacy grid has 15 bands of equal width, with solar irradiance and ozone cross section interpolated from the
48-wavelength table of `assets/spectra.json`.

The table is spectral. To convert it to colour, `sosfuse2.atmosphere.spectral.table_weights` returns weights for the CIE 1931 2°
observer (X, Y, Z) and for the scotopic luminous efficiency V′ (S′) of a given band set. A table value `L_i` is the
radiance in band `i`, and the weights assume that the ratio of radiance to solar irradiance is constant inside a band:

$$
X_c = \sum_i w_{ic}\,L_i,\qquad w_{ic} = \frac{K}{E_i}\int_{\mathrm{band}\ i}\bar c(\lambda)\,E_\odot(\lambda)\,\mathrm{d}\lambda
$$

`E_i` is the solar irradiance the solver uses for band `i` (the band mean, or the value at the band's wavelength in
`point` mode), and `K` is 683 lm W⁻¹ for X, Y and Z and 1700 lm W⁻¹ for S′.
Outside the legacy grid, the integral is evaluated exactly for the observer function and the solar spectrum taken as piecewise linear between
their nodes. For the legacy 15-band tables, `table_weights` keeps the weights those tables were made with: trapezoidal
integration over the band edges and the integer wavelengths between them, with `E_i` interpolated at the band
centre. The colour-matching functions are the CIE 1931 data (CIE 2019a), and V′ is the CIE 1951 scotopic function
(CIE 2019b). The preview images written with each table (`preview_m03.png`, `preview_m06.png`) use a separate 15-band RGB
weight set, interpolated to the band centres and scaled by each band's width (`bin_edges_nm`) over the 31.33 nm width
of the 15 reference bands. A preview of a table with one wavelength is monochrome.

## Processes not modelled

- Absorbers other than ozone, NO₂ and O₄. There is no water vapour, no absorption by the O₂ molecule and no O₂–O₂
  absorption in the A and B bands.
- Temperature dependence of the NO₂ and O₄ cross sections, NO₂ photochemistry and its change through twilight. The
  shape of the NO₂ profile is fixed, and only its column is a parameter.
- Stokes output. Polarisation is carried internally but not written.
- Polarisation by aerosol. Mie scattering matrices enter only through their scalar phase function.
- Refraction, and any effect of a finite solar disc.
- A wavelength dependence of the Rayleigh depolarisation factor of the phase function. The cross section includes it
  through the King factor.
- Wavelength dependence of the aerosol single-scattering albedo, of the asymmetry parameter unless a Mie table is
  selected, and of the ground albedo.
- Clouds, moonlight, airglow, artificial light and inelastic scattering.
- Horizontal variation of the atmosphere, surface elevation and the Earth–Sun distance.
