# Physical model

sosfuse2 computes the radiance of the clear sky seen by an observer near the ground, for a Sun that may be above the
horizon or far below it. The atmosphere is a spherical shell with horizontally uniform optical properties, the Sun is a
point source, and multiple scattering is summed to convergence. Rayleigh scattering is polarised. Aerosol scattering is
depolarising. The output is total radiance. Literature is listed in `REFERENCES.md`, data files in `THIRD_PARTY.md`.

## Output

A run writes one table per atmospheric state, `table.npz`. Its array `radiance` has the axes (solar elevation, view
elevation, relative azimuth, wavelength), is float32, and is in W m⁻² sr⁻¹ nm⁻¹. The default axes are:

| axis | values |
|---|---|
| solar elevation | +6° to −18° in 1° steps; −3° and −6° are required rows |
| view elevation | 100 rows of 0.9°, centres 89.55° to 0.45°; the horizon is the lower edge of the last row |
| relative azimuth | 180 columns of 1°, centres 0.5° to 179.5°; 0° is towards the Sun, 180° away from it |
| wavelength | 45 bands between 360 and 830 nm, centres stored as `wavelength_nm` |

The field is symmetric about the solar plane, so only one half of the sky is stored. Each value is a point evaluation
at a row and column centre, not an average over the pixel. The observer is 0.1 km above the ground. Solar elevation is
measured from the horizontal plane at the observer and is negative below the horizon.

The table holds the total radiance. Polarisation acts inside the calculation (see Rayleigh scattering below), but the
code writes no Stokes parameters, degree of polarisation or polarisation angle.

## Geometry

The Earth is a sphere of radius 6371 km with a Lambertian surface. The atmosphere ends at 80 km, and nothing lies above
it. Rays are straight lines, so refraction is not modelled. The Sun is a point at infinity, with no solar disc and no
penumbra. A point is lit when the straight line from it to the Sun does not meet the Earth, which makes the Earth's
shadow a sharp cylinder.

Because the atmosphere is uniform in the horizontal, the radiation field is symmetric about the direction to the Sun.
The solver stores it as a function of height `h` and of the angle `χ` from the sub-solar point. An observer at solar
elevation `h_s` sits at `χ = 90° − h_s`, so changing the solar elevation of a table row moves the observer along the
`χ` axis and does not change the solved field. The `χ` grid reaches 180°, so the geometry has no limit on solar
depression. The convergence test of the iteration looks at +2°, −3°, −6°, −9° and −12° only (see Iteration), and the
default table stops at −18°.

## Constituents

Six components extinguish or scatter light: air molecules, a tropospheric aerosol, ozone, NO₂, O₄ and, if enabled, a
stratospheric sulfate layer. Each extinction coefficient is the product of a vertical profile and a wavelength law (for
O₄, the square of a profile). NO₂ and O₄ were added in 1.1.0, together with a new Rayleigh law. A
state file sets the parameters; `src/sosfuse2/assets/states_v2.json` holds 64 states sampled over the axes given in the
tables below.

### Molecules

Molecular scattering follows

$$
\beta_R(h,\lambda) = \beta_{450}\,\rho_R(h)\,\frac{\sigma_B(\lambda_{\mathrm{vac}})}{\sigma_B(\lambda_{\mathrm{vac}}(450\ \mathrm{nm}_{\mathrm{air}}))}
$$

`σ_B` is the Rayleigh cross section of Bodhaine et al. (1999), their equations 18, 19, 5, 6, 23 and 22 with 360 ppm
CO₂, evaluated at the vacuum wavelength of the engine's standard-air wavelength λ (Ciddor 1996). It includes the
dispersion of the refractive index and the King factor of air, so it is not a pure power law (`src/sosfuse2/rayleigh.py`).
`β₄₅₀` is the coefficient of 1.0.0 at 450 nm, which 1.1.0 keeps: 1.0.0 used `β₅₅₀ (550 nm / λ)⁴`. Against that λ⁻⁴ law
the new coefficient is 1.037 times larger at 360 nm, equal at 450 nm, and 0.973 times at 600 nm and 0.958 times at
830 nm. The depolarisation factor of the phase function stays at 0.03 (see Rayleigh scattering below). The vertical
profile `ρ_R` depends on the density option:

| option | profile | Rayleigh optical depth, 0 to 80 km, at 450 nm / 550 nm |
|---|---|---|
| default | exp(−h / 8 km) | 0.2420 / 0.1063 (β₄₅₀ = 0.030255 km⁻¹) |
| `usstd76` with `rayleigh_column: usstd` | hydrostatic number density of the US Standard Atmosphere 1976 (NOAA et al. 1976) | 0.2215 / 0.0972 |

The second row is what the gate fixtures and `states_v1_usstd.json` use. Its optical depth at 450 nm is the sum of the
layers below 80 km in the tabulated Rayleigh optical depth at 450 nm of the IPRT phase E atmosphere (IPRT project: Emde
et al. 2015). The Rayleigh spectrum is therefore the Bodhaine law anchored at that 450 nm value. 1.0.0 used a λ⁻⁴ law
with the same anchors, which gave 0.1085 and 0.0993 at 550 nm for the two rows. The states in `states_v1.json` and
`states_v2.json` carry no density block and take the default row.

### Ozone

Ozone only absorbs. Its extinction coefficient is

$$
\beta_{\mathrm{O_3}}(h,\lambda) = \bar\sigma(\lambda)\,\frac{N}{15\ \mathrm{km}}\,q(h),\qquad \int_0^{80\,\mathrm{km}} q\,\mathrm{d}h = 15\ \mathrm{km}
$$

with the column `N` = `ozone_du` × 2.687×10²⁰ m⁻² per Dobson unit. The state axis `ozone_du` runs from 220 to 420. The
shape `q` is one of two choices. The default is a triangle, `max(0, 1 − |h − 25 km| / 15 km)`, which is non-zero between
10 and 40 km and peaks at 25 km. The `usstd76` option takes the shape from the layer absorption at 320 nm of the same IPRT
atmosphere. The density asset records that this is a shape proxy normalised to the state's column and not a measured
ozone profile.

The cross section `σ̄` comes from the Serdyuchenko and Gorshelev laboratory data (Serdyuchenko et al. 2014, Gorshelev
et al. 2014). In the default band mode of the full grid it is a mean over each band, then a mean over height weighted by
`q(h)`, with the temperature of the US Standard Atmosphere at each height and linear interpolation in temperature
between the 193 to 293 K columns. The result is one number per band, used at all heights. A fixed 233 K column can be
selected instead. The reduced smoke grid at 550 nm uses the older 48-wavelength table of `assets/spectra.json`
(Bruneton's demo data, `THIRD_PARTY.md` §3).

### NO₂

NO₂ only absorbs. Its extinction coefficient is

$$
\beta_{\mathrm{NO_2}}(h,\lambda) = \bar\sigma_{\mathrm{NO_2}}(\lambda)\,\frac{N_{\mathrm{NO_2}}}{H_f}\,f(h),\qquad H_f = \int_0^{80\,\mathrm{km}} f\,\mathrm{d}h
$$

with the column `N_NO2` = `no2_du` × 2.687×10²⁰ m⁻². The state parameter `parameters.no2_du` defaults to 0.207 DU, the
column of the reference profile, and 0 switches NO₂ off. The shape `f` is the NO₂ number density of the AFGL US Standard
profile (`afglus.dat`, Anderson et al. 1986, AFGL-TR-86-0110, as distributed with libRadtran 2.0.6), interpolated log-linearly onto 0.5 km steps
and normalised to a maximum of 1 (`assets/no2_afglus_profile.json`). It is used with every density option, also with an
external air and ozone profile. The cross section is that of Vandaele et al. (1998) at 220 K, from BIRA-IASB data
resampled to 0.1 nm cells (`assets/no2_vandaele1998_fine.json`, attribution and changes in `THIRD_PARTY.md` §11). In band
mode `σ̄` is the mean over each band, in point mode the value at the wavelength. It is the same at all heights; the
cross section has no temperature dependence.

### O₄

The collision-induced absorption of O₂–O₂ pairs (O₄) is proportional to the square of the O₂ number density:

$$
\beta_{\mathrm{O_4}}(h,\lambda) = \bar\sigma_{\mathrm{O_4}}(\lambda)\,\bigl(0.20946\,n_{\mathrm{air}}(h)\bigr)^2,\qquad n_{\mathrm{air}}(h) = n_0\,\frac{\rho_R(h)}{\rho_R(0)}
$$

`n₀` is the air number density at the surface: from the density asset for `usstd76`, 2.5469×10²⁵ m⁻³ for the default
exp(−h / 8 km) profile, whose O₄ shape is then exp(−h / 4 km), and the value that an external profile declares as
`air_surface_number_density_m3`. The state parameter `parameters.o4` (a JSON boolean, default true) switches O₄ on and
off. The cross section is the HITRAN 2024 collision-induced absorption (Gordon et al. 2026; Terragni et al. 2025): data set 54 at 293 K up to 499.861 nm, and data
set 32 at 293 K up to 600 nm and at 287 K up to 653.842 nm, resampled to 0.1 nm cells and zero outside that range. The
O₂ A- and B-band sets are excluded. `σ̄` is a band or point value as for NO₂, with no temperature dependence. The O₄ data
are not bundled. `python -m sosfuse2.prepare_minor_absorbers` prepares them once in the work root (README, Quick start;
sources and changes in `THIRD_PARTY.md` §12).

### Tropospheric aerosol

The tropospheric aerosol is described by five state parameters.

| parameter | meaning | axis in `states_v2.json` |
|---|---|---|
| `aod550` | optical depth at 550 nm, 0 to 80 km | 0.02 to 0.6, logarithmic |
| `angstrom` | exponent α of the extinction law (550 nm / λ)^α | 0.8 to 2.0 |
| `ssa` | single-scattering albedo, the same at all wavelengths | 0.85 to 0.98 |
| `scale_height_km` | `H` of the profile exp(−h / H) | 1 to 3 |
| `g` | asymmetry parameter of the Henyey–Greenstein phase function, the same at all wavelengths | 0.65 to 0.8 |

The extinction coefficient is `aod550 · exp(−h/H) / ∫₀⁸⁰ exp(−h/H) dh · (550 nm / λ)^α`, and the scattering coefficient is
`ssa` times that. The phase function is that of Henyey and Greenstein (1941),
`p(cos Θ) = (1 − g²) / [4π (1 + g² − 2g cos Θ)^{3/2}]`.

Two options replace parts of this default. A state block `troposphere_phase` swaps the Henyey–Greenstein function for a
tabulated Mie phase function and leaves extinction, albedo and Ångström law as set by the state. The shipped table is a
bimodal lognormal rural aerosol at 70 % relative humidity, built from the Shettle and Fenn (1979) rural model with the
mode parameters checked against LOWTRAN 7 tables (Kneizys et al. 1988). It is computed with Bohren–Huffman spheres
(Bohren and Huffman 1983) at 18 wavelengths between 360 and 830 nm and interpolated linearly in wavelength. A state
block `troposphere_profile` replaces the single exponential by a boundary layer and a free troposphere joined by a
smooth step, centred at 2 km by default. No shipped state file carries either block.

### Stratospheric sulfate layer

The layer is off unless a state marks it `enabled` or the runner is given `--stratosphere on`. `states_v1.json` and
`states_v2.json` mark it `unsupported`, `states_v1_usstd.json` marks it `disabled`, and the test fixtures enable it in
some states. When on, its profile is a Gaussian, by default centred at 21 km with σ = 3 km and normalised over 0 to
80 km, and its optical depth at 550 nm is `stratosphere_tau550` (axis 0.002 to 0.006 in `states_v2.json`). The particles
are a single lognormal sulfate population with median radius 0.08 µm, geometric width 1.6 and refractive index 1.43 + 0i,
so they do not absorb. The shipped Mie table covers 360 to 830 nm at 18 wavelengths. Extinction at other wavelengths
scales with the Mie cross section relative to 550 nm. The asymmetry parameter of the table falls from 0.713 at 360 nm to
0.510 at 830 nm. The layer uses its own tabulated phase function in place of Henyey–Greenstein.

### Ground

The surface is Lambertian with an albedo `albedo` (axis 0.05 to 0.8) that does not depend on wavelength. It reflects the
direct solar beam and the diffuse radiation reaching it, and the reflected light is scattered by the atmosphere again.
There is no topography and no directional reflectance.

## Scattering

### Rayleigh scattering and polarisation

For intensity, Rayleigh scattering uses the phase function

$$
p(\Theta) = \Delta\,\frac{3}{16\pi}\left(1 + \cos^2\Theta\right) + \frac{1-\Delta}{4\pi},\qquad \Delta = \frac{1-\delta}{1+\delta/2}
$$

with a constant depolarisation factor δ = 0.03, so Δ = 0.9557. The radiance field is carried as a symmetric 3×3
coherency tensor. Its trace is the intensity. Reflection symmetry about the solar plane leaves four independent
components, which the solver stores in a local frame at each cell. Molecular scattering acts on the full tensor, as dipole
scattering with the depolarisation factor δ, so polarisation changes the intensity at every order. The runner option
`--polarization-model` accepts `preserving`, but only the `depolarizing` model is implemented and the other value stops
the run with `E_UNSUPPORTED_POL_MODEL`.

### Aerosol scattering

Tropospheric aerosol and stratospheric sulfate scatter with scalar phase functions and depolarise completely. The
scattered light is unpolarised, with the intensity given by the scalar phase function, whatever the polarisation of the
incident light. The tabulated aerosol phase functions are the first element of the Mie scattering matrix. The other
elements are not used in the transport.

### Truncation of phase functions

The direct beam is scattered once with the full phase function. For tropospheric aerosol that is the Henyey–Greenstein
function or the tabulated Mie function, and for the stratosphere it is the Legendre series of the Mie phase function to
degree 32. In the multiple-scattering source, the phase function of each aerosol is written as
`p(cos Θ) = (1/4π) Σ χ_l P_l(cos Θ)` and cut at degree `L`, which is 16 in the production setting and may be set to 24
or 32. The coefficients are `χ_l = (2l+1) g^l` for Henyey–Greenstein and the tabulated moments for Mie. There is no delta-M
or other renormalisation of the forward peak, so the truncated terms are dropped. Rayleigh scattering needs no
expansion, because its phase function is a quadratic polynomial in direction and the four-component tensor holds it
exactly.

## Solution

### Single scattering

Sunlight scattered once into the line of sight is integrated along that line. The
optical depth from the Sun to the scattering point comes from a precomputed table of slant columns of each constituent
along the solar direction, as a function of height and angle. The optical depth along the line of sight is integrated
with Simpson's rule between the quadrature nodes. Lines of sight are cut where they cross the 10, 25 and 40 km spheres,
the Earth's limb, the shadow boundary and the ground. Each piece carries 12 Gauss–Legendre nodes, or 24 for the lines of
sight of the output table. Light reflected once by the ground is part of the first order and is scattered with the
untruncated phase functions.

### Multiple scattering

Higher orders are carried by a source field defined on the (height, `χ`) grid. In each cell the source is stored as the
four Rayleigh tensor components plus Legendre-expansion coefficients of the aerosol scattering source, with a second set
for the stratosphere if it is enabled. One application of the transport operator works as follows. For each cell, the
radiance arriving from every incident direction is computed by integrating the source field along that direction, with
attenuation by all constituents and with the light reflected from the ground where the ray ends on it. The source is
interpolated linearly in height and `χ` and then rescaled by `(G/A)^W`, where `G` and `A` are the weighted geometric and
arithmetic means of the corner traces that are at least `FLT_MIN` and `W` is their total weight. With all corners valid the
trace is the geometric mean; as `W` goes to 0 the result goes continuously to the linear interpolation.
The arriving radiance is then projected onto the source basis, which gives the next source
field. The incident directions are Gauss–Legendre nodes inside intervals whose ends are the limb, the local horizon,
the shadow boundary and the directions tangent to the shells at 0, 1.2, 8, 25, 40 and 80 km.

### Iteration

The source field is the fixed point of this operator. One application adds one order of scattering, so plain iteration
from zero is the successive-orders method. The production runner instead accelerates the iteration with Anderson mixing
(Anderson 1965) of depth 4, applied to each wavelength separately. A wavelength has converged when, from the second
evaluation onward, both the change of the source field and the change of the observer radiance between successive
iterates are at most 10⁻³ in relative terms. The observer test uses 35 radiances: seven viewing directions for each of
the five solar elevations +2°, −3°, −6°, −9° and −12°, each averaged over a 2° × 2° box. The runner stops with
`E_NOT_CONVERGED` if a wavelength has not converged within the evaluation limit, 64 by default. Iterates are not
scattering orders, so the number recorded in `meta.json` is a count of evaluations.

When the source field has converged, the observer radiance for every row of the table is computed by integrating the
converged source and the single-scattering term along the lines of sight. All solar elevations use the same source field.

### Implementation

A NumPy reference reproduces the arithmetic of the CUDA kernel in single precision, with double-single arithmetic for
the ray geometry, and is the golden reference of the numeric gates. Tolerances and gates are described in the README and
in `docs/CHECKS.md`. The solver definitions in `src/sosfuse2/solver_*.py` define the grids, the quadrature and the
physical constants, and `src/sosfuse2/rayleigh.py` and `minor_absorbers.py` hold the Rayleigh and minor-absorber
coefficients of 1.1.0.

## Production grid

| quantity | value |
|---|---|
| heights | 44 nodes from 0 to 80 km: 20 base nodes (0, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2, 3, 4.5, 6.5, 9, 13, 18, 26, 40, 60, 80 km), their midpoints below 13 km, and two interior thirds of each interval above 13 km |
| `χ` | 75 nodes: 10° steps to 80°, 1° to 85°, 0.5° from 85° to 109.5°, 2° to 118°, then 10° steps from 120° to 180° |
| incident directions per cell | 12 Gauss–Legendre nodes per polar interval, 16 azimuth intervals over 0° to 180° |
| line-of-sight nodes | 12 per piece for the source, 24 per piece for the output |
| Legendre degree `L` | 16 |
| solar-column table | resolution scale 4 |
| Anderson tolerance, depth | 10⁻³, 4 |

The reduced option of the runner uses 4 heights, 10 angles and one wavelength (550 nm). It is a software check and its
grid is too coarse for physical use.

## Spectral treatment and colour

The default wavelength grid has 45 bands between 360 and 830 nm. It starts from 40 bands of 11.75 nm, and five of them
(489.25 to 501, 512.75 to 524.5, 524.5 to 536.25, 559.75 to 571.5 and 642 to 653.75 nm) are each split into two halves.
The solver runs at the centre wavelength of each band. The solar spectral irradiance and the ozone, NO₂ and O₄ cross
sections are band means taken from a 0.1 nm grid. Rayleigh and aerosol coefficients, and the Mie tables, are evaluated at the band centre
and not averaged over the band.

The solar spectrum is the TSIS-1 Hybrid Solar Reference Spectrum version 2 (Coddington et al. 2021, 2023), averaged to
0.1 nm cells, multiplied by 0.999061 to match the total solar irradiance measured by TSIS-1 (the factor is recorded in
`spectra_fine.json`), and relabelled from vacuum to standard-air wavelengths with the equations of Ciddor (1996). The
ozone data carry the same relabelling. No correction for the distance between the Earth and the Sun is applied. The older
legacy grid has 15 bands of equal width, with solar irradiance and ozone cross section interpolated from the
48-wavelength table of `assets/spectra.json`.

The table is spectral. For colour, `sosfuse2.spectral.table_weights` returns weights for the CIE 1931 2° observer (X, Y, Z)
and for the scotopic luminous efficiency V′ (S′) of a given band set. A table value `L_i` is the radiance in band `i`, and
the weights assume that the ratio of radiance to solar irradiance is constant inside a band:

$$
X_c = \sum_i w_{ic}\,L_i,\qquad w_{ic} = \frac{K}{E_i}\int_{\mathrm{band}\ i}\bar c(\lambda)\,E_\odot(\lambda)\,\mathrm{d}\lambda
$$

Here `E_i` is the band-mean solar irradiance used by the solver, and `K` is 683 lm W⁻¹ for X, Y and Z and 1700 lm W⁻¹
for S′. The integral is evaluated exactly for the observer function and the solar spectrum taken as piecewise linear
between their nodes. The colour-matching functions
are the CIE 1931 data (CIE 2019a), and V′ is the CIE 1951 scotopic function (CIE 2019b). The preview images written with
each table (`preview_m03.png`, `preview_m06.png`) use a separate 15-band RGB weight set, interpolated to the band
centres.

## Not in the model

- Absorbers other than ozone, NO₂ and O₄. There is no water vapour, no absorption by the O₂ molecule and no O₂–O₂
  absorption in the A and B bands.
- Temperature dependence of the NO₂ and O₄ cross sections, NO₂ photochemistry and its change through twilight. The
  shape of the NO₂ profile is fixed and only its column is a parameter.
- Stokes output. Polarisation is carried internally but not written.
- Polarisation by aerosol. Mie scattering matrices are used only through their scalar phase function.
- Refraction, and any effect of a finite solar disc.
- A wavelength dependence of the Rayleigh depolarisation factor of the phase function. The cross section includes
  it through the King factor.
- Wavelength dependence of the aerosol single-scattering albedo, of the asymmetry parameter unless a Mie table is
  selected, and of the ground albedo.
- Clouds, moonlight, airglow, artificial light and inelastic scattering.
- Horizontal variation of the atmosphere, surface elevation and the Earth–Sun distance.
