# Validation

The research records document the checks below. The first radiance comparison with MYSTIC (Figure 1) predates 1.1.0 and uses
one atmospheric state, called state A in this document, and one sky pixel: the sun side of the sky at 20.25° elevation and
0.5° relative azimuth, seen from 0.1 km above the ground, at solar zenith angles (SZA) of 93°, 96° and 99°, which are solar
elevations −3°, −6° and −9°. State A has a tropospheric aerosol with aod550 0.0551, Ångström exponent 1.889,
single-scattering albedo 0.898, Henyey–Greenstein asymmetry parameter 0.683 and a 1.0 km scale height; an ozone column of
224.75 DU; a Gaussian stratospheric sulfate layer with centre 19 km, σ 3 km and τ550 0.0037; and a Lambert surface of albedo
0.376. The checks of the 1.1.0 absorbers add a second state and two more directions, the zenith and the anti-solar sky. We
have not yet compared the engine's output with a measured sky.

![Engine and MYSTIC radiance spectra and their ratio at SZA 93, 96 and 99 degrees](validation/mystic_vs_engine.png)

Figure 1. Top: radiance divided by the solar irradiance, L/E₀, from the engine (lines) and from MYSTIC (open markers). The
relative standard error of MYSTIC lies between 0.52 % and 2.41 % and is not drawn. Bottom: MYSTIC divided by the engine, with
the MYSTIC standard error as bars; the three SZA are offset by ±3 nm so that the bars do not overlap. Data:
`docs/validation/mystic_compare.csv`. Script: `docs/validation/plot_validation.py`.

## Same-input comparison with MYSTIC (1.0.0 physics)

The reference is MYSTIC, the Monte Carlo solver of libRadtran 2.0.6 (Mayer 2009; Emde et al. 2016), run in a 1D
spherical-shell atmosphere with polarisation and backward tracing. Both codes were given the same optical input, so any
difference between them comes from the transport and its numerics rather than from the atmosphere.

The input is the engine's, written out for MYSTIC in 0.5 km layers up to 80 km plus the observer level at 0.1 km: Rayleigh
and ozone optical depth per layer, the tropospheric and stratospheric aerosol with their phase functions, and the surface
albedo. The Earth radius is 6371 km and the Rayleigh depolarisation factor is 0.03 in both codes. Aerosol scattering
depolarises completely in both, in the engine by construction and in MYSTIC because it is given only the P₁₁ element of the
phase matrix, and neither code models refraction. A script rebuilds the engine's coefficients from the same asset files and
state parameters and writes the MYSTIC input without calling the engine, so a mistake in that script would appear as a
transport difference. Reading the generated files back through a DISORT direct-beam calculation reproduces the aerosol
optical depth to a relative error of at most 5.8e-7 and the total vertical optical depth to at most 3.7e-7, at 430.5, 555.8
and 696.8 nm.

The engine radiance comes from a table baked with the research build. That table uses the legacy spectral inputs, the 48-band
solar and ozone table `src/sosfuse2/assets/spectra.json` (the shipped file has the sha256 recorded for the comparison), read
as point values at 30 wavelengths, the centres of 30 equal bins over 360–830 nm, and MYSTIC ran at the same 30 wavelengths as
single wavelengths. The 1.0.0 default on the production grid differs: band means and a temperature-dependent ozone cross
section (`src/sosfuse2/spectral.py`). The ozone cross section of the legacy table has a half-bin shift that was corrected in
the research build. Because both codes read the same legacy table, that shift does not affect the comparison, but the
comparison also does not test the current spectral defaults, and we did not re-bake the table with the installed package for
it. The research build had the 1.0.0 physics, a λ⁻⁴ Rayleigh law and no NO₂ or O₄, and the MYSTIC input was made from the
same coefficients, so this comparison tests the transport of the 1.0.0 model. The 1.1.0 absorbers are tested below, under
*NO₂ and O₄ in 1.1.0: engine against MYSTIC on the same input*.

Each of the 90 cells (3 SZA × 30 wavelengths) combines two or four independent MYSTIC runs with different random seeds, 260
runs and 2,074,640,958 photons in total. The first pass was too noisy at SZA 99 and in 10 of 30 cells at SZA 96, so those
cells received further runs. The cell value is the mean over its runs weighted by photon count, and its standard error (SE)
is the photon-weighted combination of the standard errors that MYSTIC reports for each run (`mc_std`). Each SZA has a target
for the relative standard error of every cell: 1 % at SZA 93 and 96, 2.5 % at SZA 99.

The ratio in Table 1 is MYSTIC divided by the engine. Blue is the mean ratio over the wavelengths below 450 nm and red the
mean over those above 600 nm. The last column counts the cells whose MYSTIC standard error meets the target; it describes the
precision of the MYSTIC value, not the agreement between the two codes.

| SZA | ratio min / median / max | blue / red mean ratio | MYSTIC relative SE | cells within target |
|---:|---|---|---|---|
| 93° | 0.9844 / 0.9979 / 1.0102 | 0.9971 / 0.9999 | 0.52–0.77 % | 30 of 30 (1 %) |
| 96° | 0.9890 / 1.0016 / 1.0217 | 1.0017 / 1.0003 | 0.68–0.97 % | 30 of 30 (1 %) |
| 99° | 0.9584 / 1.0040 / 1.0634 | 1.0027 / 1.0105 | 1.97–2.41 % | 30 of 30 (2.5 %) |

Table 1. Radiance ratio MYSTIC / engine at the three solar zenith angles.

We compare the colour through the CIE 1976 coordinates u′ and v′ of the 30-wavelength spectrum, weighted with the solar
irradiance and the CIE 1931 colour-matching functions. Δ is MYSTIC minus engine. The standard error comes from 200 random
draws of the MYSTIC spectrum within its per-wavelength standard errors, with the engine spectrum held fixed. The draws treat
wavelengths as independent, which holds because every cell has its own random seeds.

| SZA | engine u′, v′ | MYSTIC u′, v′ | Δu′ ± SE | Δv′ ± SE |
|---:|---|---|---|---|
| 93° | 0.203054, 0.466584 | 0.203050, 0.466542 | −0.000004 ± 0.000181 | −0.000042 ± 0.000252 |
| 96° | 0.189567, 0.435841 | 0.189052, 0.435705 | −0.000514 ± 0.000316 | −0.000135 ± 0.000496 |
| 99° | 0.191249, 0.432994 | 0.192429, 0.433041 | +0.001181 ± 0.000754 | +0.000047 ± 0.001215 |

Table 2. Chromaticity of the sun-side pixel in both codes.

At SZA 93 and 96 the ratio of every cell lies within 2.2 % of 1, with a standard error below 1 % per cell. At SZA 99 the
ratios range over 0.958–1.063 with a standard error of 2.0–2.4 % per cell. The largest colour differences are Δu′ = −0.0005 ±
0.0003 at SZA 96 and +0.0012 ± 0.0008 at SZA 99, both 1.6 times their standard error. Δu′ changes sign over the three angles
(−, −, +), and Δv′ stays within one standard error at all three, so at the stated precision we detect no systematic
difference. For SZA 99 the largest cell deviation |ratio − 1| / SE is 2.69; one of the 30 cells exceeds 2 and none exceeds 3,
where 30 independent normal deviates would give 1.4 cells above 2.

At SZA 99 the per-cell target changed from 2 % to 2.5 %. The estimated top-up for 2 % was 6.05 CPU-hours, whereas 2.5 %
needed an estimated 3.64 CPU-hours. The first pass reached 2 % in none of the 30 cells, with relative standard errors of
3.9–18.4 %. After the top-up, 2 of 30 cells meet 2 % and 30 of 30 meet 2.5 %.

The standard errors are Monte Carlo errors of MYSTIC only and contain no uncertainty of the engine, of the layer
discretisation or of the input. Because both codes read the same input, an error in the input itself, whether in the ozone
cross section, the Mie table, the solar spectrum, the colour-matching functions or the Rayleigh optical-depth law, is common
to both and cancels in the ratio, so the comparison cannot detect such errors. It covers one state, one pixel and three SZA;
we did not run other directions, other states or solar depressions beyond 9°.

## MYSTIC build check

The comparison depends on the MYSTIC build and its spherical 1D set-up. To check them, we ran the same libRadtran build on
the Rayleigh benchmark of Korkin et al. (2020): a spherical-shell atmosphere with conservative Rayleigh scattering of optical
depth 0.25 and 1, a black surface, and the radiance seen at the top of the atmosphere, for which the paper tabulates MYSTIC
results. We reproduced 120 of those cells, covering optical depths 0.25 and 1, solar zenith angles 0°, 70.47° and 84.26°,
relative azimuths 0° and 180°, and single-scattering and total radiance, with 30 cells for each combination of optical depth
and mode. Each cell used 1,690,000 photons and one random seed. The `uvspec` binary has sha256
`d0c5ff763780e13455a95f4eaaa7c5ea936d9f1cbdf1d3eeb4c2a9439b42abb8`, which is also the hash of the binary of the tree that
ran the comparison above.

The largest absolute deviation from the tabulated MYSTIC value is 0.2728 %, and the mean absolute deviation is 0.0643 %. No
cell deviates by more than 3 standard errors (0 of 120). Because the paper gives Monte Carlo accuracies per group of cells
rather than per cell, the standard error combines the error of our run with the accuracy stated for each group. The benchmark
values themselves are not reproduced in this repository.

Three limits apply. The paper's vertical structure was not available as a table, so we tried two candidates on three test
cells, a uniform shell from 0 to 100 km and an exponential profile with an 8 km scale height. Only the uniform shell
reproduced the table; the exponential profile was 4.2 % and 5.7 % off at two of the cells and 99.99 % low at the third, so
the structure was chosen by matching the table. The benchmark neglects polarisation and has no aerosol, no absorption and no
reflecting surface, so this check tests neither MYSTIC's polarised mode nor any of the optical input of the comparison above.
Finally, its geometry is a top-of-atmosphere view rather than a sky seen from the ground.

## Stratospheric aerosol input and zenith polarisation (Ugolnikov et al.)

This calculation was made with MYSTIC rather than with the engine. We gave MYSTIC the engine's optical input for state A,
including its stratospheric sulfate layer, and read the Stokes vector it returns. The engine itself has no Stokes output,
since its tables hold a single radiance per direction and wavelength, so this is a qualitative check of the input, namely the
stratospheric sulfate layer, and does not test the engine's transport.

The calculation looks straight up at SZA 90–99 and 30 wavelengths and takes the degree of linear polarisation, DoLP = √(Q² +
U²)/I. Additional runs at SZA 93–96 and 5 wavelengths reduced the standard error where the features described below appear.
The standard error of a DoLP value is 0.001–0.006 at SZA 91–94, 0.006–0.011 at SZA 95 and 0.011–0.022 at SZA 96.

The reference is published measurements of the polarisation of the zenith twilight sky in a V band (550 nm) and an R band
(700 nm) (Ugolnikov et al. 2004). In the V band the measured polarisation stays nearly constant until the solar elevation
reaches about −5° and then decreases. In the R band the measurements show day-to-day differences in polarisation in bright
twilight down to about −3°, after which the differences nearly vanish, and a fall after −5° from multiple scattering. Later
papers by Ugolnikov and Maslov relate twilight-sky signals to stratospheric aerosol: a scattering effect that grows with
wavelength up to 35 km (Ugolnikov and Maslov 2018a) and a depolarisation near 20 km (Ugolnikov and Maslov 2018b).

Two features are reproduced. The first is a decrease of the polarisation after SZA 95: at 555.8 nm the model DoLP is
0.71–0.73 from SZA 90 to 95 and 0.66 ± 0.01 at SZA 96, while the values at SZA 97 and beyond are too noisy to read (0.60 ±
0.06 at SZA 97). The second is an aerosol-caused reduction of the R-band polarisation near the solar elevations where the
measurements show day-to-day differences. In the model it appears as a dip with a recovery. At 696.8 nm the DoLP is 0.776 at
SZA 91, 0.725 at SZA 93, 0.712 ± 0.003 at SZA 94, 0.755 ± 0.007 at SZA 95 and 0.695 at SZA 96. At 743.8 nm the dip is deeper:
0.782 at SZA 91, 0.700 ± 0.003 at SZA 94 and 0.769 ± 0.007 at SZA 95. The measurements describe a reduction that depends on
the day rather than a dip, so what is reproduced is the presence of the reduction in this range of solar elevation, not its
shape.

To find the cause, we removed the stratospheric aerosol by setting its column to zero with everything else equal and repeated
the calculation for SZA 91–96 at 555.8, 696.8 and 743.8 nm. Without the layer the dip and the recovery are gone: at 743.8 nm
the DoLP is then 0.840, 0.861, 0.859, 0.849, 0.820 and 0.774 for SZA 91–96, a maximum near SZA 92–93 followed by a steady
decrease. The depolarisation by the layer, DoLP with the layer minus DoLP without it, is −0.058, −0.086, −0.132, −0.149 and
−0.051 at 743.8 nm for SZA 91–95. At 555.8 nm it is at most −0.073 (SZA 93), about half the largest value at 743.8 nm. The
direction agrees with the measurements, in which the aerosol effect is larger at longer wavelengths.

We made no quantitative comparison of the absolute polarisation or of the depth of the dip: the published values exist only
as figures, and the observing sites, dates and aerosol loads differ from state A. In this calculation the aerosol scatters
with P₁₁ only and therefore depolarises completely, as in the engine; we did not use a full Mie phase matrix and did not test
what that assumption does to the depth of the dip. We have not inverted the stratospheric optical depth from measured
polarisation with the relation of Ugolnikov and Maslov (2019) or compared it with the input value of 0.0037.

## Input physics added in 1.1.0

Up to 1.0.0 the engine had no NO₂ and no O₄ absorption, and it scaled the Rayleigh optical depth as a pure power law in
wavelength. These are choices of input physics rather than errors of the solver, and the same-input comparison above cannot
detect them because both codes read the same input. We measured them instead against libRadtran's own molecular input, by
running MYSTIC twice, with and without the item, on otherwise identical input. The results led to the three changes of 1.1.0:
NO₂ and O₄ absorption and the Rayleigh cross section of Bodhaine et al. (1999) (`docs/MODEL.md`, *Constituents*). The numbers
in this section are differences between two MYSTIC runs with libRadtran's own cross sections and profiles, not between MYSTIC
and the engine; the 1.0.0 engine corresponds to the run without the item. For scale, the research records judge changes of
the solver against a tolerance of Δu′v′ 2.9e-4 at the 95th percentile and 0.122 % in the RMS of the luminance. All three
effects were measured for one state: the pair runs for O₄ and Rayleigh at SZA 96 and the NO₂ runs at SZA 93–99, in the
directions listed below. Band-mean spectra, other states and other directions were not measured with libRadtran's own input.

### NO₂

libRadtran's `afglus` profile contains NO₂ with a column of 5.57e15 cm⁻² (0.207 DU), a vertical optical depth of about 3e-3
between 370 and 430 nm. To measure its effect, we gave two MYSTIC runs that differ only in NO₂ the same random seeds, so that
their photon paths are nearly identical and the ratio is precise. Each pixel in Table 3 has 8 seeds of 500,000 photons at
each of 30 wavelengths. The ratio is with NO₂ divided by without, and Δ is with minus without. ΔY is the relative change of
the luminance Y, with the same weights as the colour above.

| view | SZA | blue mean ratio | ΔY | Δu′ | Δv′ |
|---|---:|---:|---|---|---|
| sun side 20° | 93° | 0.927 | −1.40 ± 0.01 % | +0.00117 ± 0.00001 | +0.00351 ± 0.00002 |
| sun side 20° | 96° | 0.924 | −1.83 ± 0.03 % | +0.00107 ± 0.00003 | +0.00480 ± 0.00006 |
| sun side 20° | 99° | 0.900 | −2.07 ± 0.12 % | +0.00133 ± 0.00012 | +0.00624 ± 0.00042 |
| zenith | 93° | 0.926 | −1.59 ± 0.01 % | +0.00094 ± 0.00001 | +0.00469 ± 0.00003 |
| zenith | 96° | 0.917 | −2.15 ± 0.04 % | +0.00086 ± 0.00006 | +0.00600 ± 0.00014 |
| zenith | 99° | 0.914 | −1.2 ± 1.4 % | +0.0005 ± 0.0018 | +0.0092 ± 0.0013 |
| anti-solar 10° | 93° | 0.920 | −1.67 ± 0.02 % | +0.00133 ± 0.00002 | +0.00454 ± 0.00005 |
| anti-solar 10° | 96° | 0.919 | −2.11 ± 0.02 % | +0.00107 ± 0.00003 | +0.00550 ± 0.00018 |

Table 3. Effect of the `afglus` NO₂ column in paired MYSTIC runs.

In all eight pixels NO₂ lowers the blue radiance and raises v′. The 1.0.0 engine, without NO₂, therefore gave a lower v′, a
bluer sky, than the same sky with this NO₂ column, by 0.0035 to 0.0092, which is 12 to 32 times the colour tolerance quoted
above. The change grows with SZA and varies by about ±25 % with direction at a given SZA. At the zenith at SZA 99 the blue is
so dark that rare high-order scattering paths dominate and the two runs decorrelate, which is why the errors are large there,
but Δv′ is still 7 times its standard error. Two independent runs with different seeds at the sun-side pixel at SZA 96 give
Δv′ = +0.0049 ± 0.0006, consistent with the paired value of +0.00480 ± 0.00006.

These runs use the column of one profile. Stratospheric NO₂ varies by a factor of several with season, latitude and twilight
chemistry, so the size of the effect for a given sky does not follow from them; we expect it to scale roughly with the column
but did not measure that. 1.1.0 takes the column as a state parameter, `parameters.no2_du`, with 0.207 DU as the default.

### O₄

libRadtran includes the O₄ absorption computed from the O₂ profile, with a vertical optical depth of 8.3e-3 at 477.5 nm, its
maximum, and 7.4e-3 at 571.5 and 634.2 nm. At the sun-side pixel at SZA 96, adding O₄ to MYSTIC changes the radiance by
factors of 0.966, 0.939, 0.964 and 0.913 at 477.5, 571.5, 587.2 and 634.2 nm, and by about 1 at the other wavelengths. The
colour changes by Δu′ = −0.0013 ± 0.0004 and Δv′ = −0.0007 ± 0.0006. These are two independent runs with two seeds of 9
million photons per wavelength. Because the 30 wavelengths are point samples that hit the narrow O₄ bands at their centres,
we expect a smaller effect on band-mean spectra, which we did not measure with MYSTIC; the engine's own change on the band
grid is given under *Size of the 1.1.0 changes on the production grid*.

### Rayleigh optical-depth law

The 1.0.0 engine took the Rayleigh optical depth at 550 nm from the IPRT 450 nm table for the US Standard Atmosphere
multiplied by (450/550)⁴, and scaled it with wavelength as (550/λ)⁴. The Rayleigh optical depth of Bodhaine et al. (1999),
which the libRadtran run used (`crs_model rayleigh Bodhaine`), is not an exact λ⁻⁴ law. For libRadtran's `afglus` atmosphere
the ratio of the Bodhaine optical depth to the 1.0.0 engine's is 1.031 at 368 nm, 1.000 at 446 nm and 0.957 at 822 nm,
measured as vertical optical depth in a DISORT direct-beam calculation.

Scaling the engine's Rayleigh optical depth wavelength by wavelength to the Bodhaine value changes the MYSTIC radiance at the
sun-side pixel at SZA 96 by an inverse-variance weighted mean ratio of 0.9892 ± 0.0020, 5.4 standard errors from 1. The ratio
is 0.9825 ± 0.0033 below 420 nm and above 700 nm, and 0.9917 ± 0.0030 between 450 and 650 nm. The colour does not move beyond
its standard error, with Δu′ = +0.0001 ± 0.0004 and Δv′ = +0.0001 ± 0.0006. Only the optical depth was changed; the
depolarisation factor of 0.03 was kept.

1.1.0 uses the Bodhaine cross section, evaluated at the vacuum wavelength, and keeps the 1.0.0 coefficient at 450 nm
(`src/sosfuse2/rayleigh.py`). Its ratio to the 1.0.0 coefficient is 1.032 at 368 nm, 1 at 450 nm and 0.958 at 822 nm, close
to the libRadtran ratios above. The depolarisation factor of the phase function stays at 0.03.

## NO₂ and O₄ in 1.1.0: engine against MYSTIC on the same input

Here we test whether the new absorbers change colour and luminance by the same amount in the engine as in MYSTIC when both
read the same optical input. Both codes read the same absorption optical depth in every layer, so the test covers how the
absorption is carried through the spherical shell, through slant paths, the Earth's shadow and multiple scattering, rather
than the cross sections or the profiles themselves. The package tests compare the cross-section assets with the original data
files when those files are supplied (`SOSFUSE2_BIRA_DIR`, `SOSFUSE2_HITRAN_CIA`; README, *Tests and checks*).

The engine was the 1.1.0 development build with the new physics. It made CUDA tables of two states with `spectral_mode`
`point` at 30 wavelengths (367.8–822.2 nm) and the ozone cross section at a fixed 233 K. We did not use the default band
mode, because MYSTIC runs at single wavelengths cannot reproduce band means, and both codes used the solar irradiance and the
colour weights of this point set-up. The engine runs were made before the 1.1.0 change of the interpolation rule of the
transport (`docs/MODEL.md`, *Multiple scattering*) and were not repeated after it. They used the production interpolation
grid, on which that change is small: in the full-grid gate state it moves the radiance by at most 3.7e-6 relative (*Size of
the 1.1.0 changes on the production grid*), far below the tolerance of 1e-4 used here. The two states of this check were not
re-solved.

The states are A, described at the top of this document, and B. State B has aod550 0.168, Ångström exponent 1.62,
single-scattering albedo 0.954, asymmetry parameter 0.753 and a 2.53 km scale height; an ozone column of 307.4 DU; a
stratospheric sulfate layer with τ550 0.0022 (centre 19 km, σ 3 km); and a surface albedo of 0.559. Both states use the
`usstd76` density and ozone profiles. There are 12 pixels. In state A they are the sun side (20.25°, 0.5°) and the zenith at
SZA 93, 96 and 99, and the anti-solar direction (10°, 180°) at SZA 93 and 96. In state B they are the sun side and the
anti-solar direction at SZA 94 and 96.

We ran four variants of each state: 00 with neither absorber (the Bodhaine Rayleigh law only), 10 with NO₂ (0.207 DU), 01
with O₄, and 11 with both. The change Δ of a variant is its value minus that of 00 for u′ and v′, and Y_variant/Y_00 − 1 for
the luminance Y.

The MYSTIC input is the engine's optical input written per layer, as in the first comparison. In a ledger of 960 layer
entries the molecular optical depths agree with the engine's to a relative 1.8e-15, and a DISORT direct-beam run on the
generated files reproduces the total vertical optical depth to 3.9e-7. The four variants share every input file except the
molecular absorption file. Every MYSTIC run of a variant used the same random seed as a run of 00, with 500,000 photons per
seed and wavelength, so the two runs follow nearly the same photon paths and their difference is far more precise than that
of independent runs. Each comparison started with 8 seeds; unresolved comparisons were doubled to 16 seeds, and the zenith
pixel at SZA 96 to 32. In total there were 23,040 runs and 35.2 CPU-hours of `uvspec` time.

The criterion compares D = Δ_engine − Δ_MYSTIC with a tolerance of max(0.1 |Δ_MYSTIC|, 1e-4): the engine has to reproduce the
change to 10 % of its size, or to 1e-4 in u′, v′ or Y_rel when the change is small. The 95 % Student t interval of D comes
from the per-seed paired changes and contains only the sampling error of MYSTIC. A value passes when the interval lies inside
±tolerance, fails when it lies wholly outside, and is unresolved otherwise.

| variant | MYSTIC Δu′ | MYSTIC Δv′ | MYSTIC ΔY | pass | fail | unresolved |
|---|---|---|---|---:|---:|---:|
| NO₂ (10) | +0.0003 to +0.0014 | +0.0037 to +0.0077 | −1.4 to −2.6 % | 30 | 0 | 6 |
| O₄ (01) | −0.0007 to −0.0022 | −0.0005 to −0.0015 | −1.0 to −2.8 % | 13 | 0 | 23 |
| both (11) | −0.0010 to +0.0002 | +0.0033 to +0.0065 | −3.0 to −4.9 % | 23 | 0 | 13 |

Table 4. Outcome of the paired comparison over the 12 pixels.

Of the 108 values (12 pixels × 3 variants × 3 quantities), 66 pass, none fails and 42 are unresolved. By quantity, Y passes
in 30 of 36 cases, u′ in 19 and v′ in 17. An unresolved value means that the MYSTIC interval stayed wider than the tolerance
after the top-ups; it is not evidence of a difference. Most unresolved values are colour changes from O₄, which are so small
that the tolerance is the 1e-4 floor. Where the MYSTIC change has a relative standard error of 2 % or less (41 of the 108
values), the engine's change divided by MYSTIC's lies between 0.971 and 1.048. Every value is listed in
`docs/validation/b2_tier1_effects.csv`.

With 16 seeds the zenith pixel at SZA 96 gave one failure, in which the engine's O₄ change of v′ was 0.72 times MYSTIC's.
With 32 seeds MYSTIC gives Δv′ = −0.00100 [−0.00121, −0.00080] against −0.00094 in the engine, so the value becomes
unresolved and the Y change passes. The 16-seed interval had been too narrow: in deep twilight a t interval from 16 paired
seeds can understate the scatter, and the 16-seed values there should be read with that in mind.

The 00 runs also give an absolute same-input comparison of the 1.1.0 engine without NO₂ and O₄, at two states and 12 pixels.
It is not part of the criterion. The engine values were read at the nearest table pixel, which on the sun side is the MYSTIC
direction itself; at the zenith the pixel is at 89.55°, and in the anti-solar direction at 10.35° elevation and 179.5°
azimuth. The standard errors in Table 5 come from the scatter of the seed values.

| state | view | SZA | seeds | MYSTIC/engine per wavelength, min / median / max | Y MYSTIC/engine | Δu′ (MYSTIC − engine) | Δv′ |
|---|---|---:|---:|---|---|---|---|
| A | sun side | 93° | 16 | 0.993 / 0.999 / 1.004 | 0.9988 ± 0.0004 | −0.00001 ± 0.00005 | +0.00015 ± 0.00009 |
| A | sun side | 96° | 16 | 0.970 / 0.999 / 1.022 | 0.9987 ± 0.0031 | +0.00000 ± 0.00035 | +0.00094 ± 0.00065 |
| A | sun side | 99° | 16 | 0.939 / 1.019 / 1.079 | 1.0049 ± 0.0196 | +0.00002 ± 0.00130 | −0.00096 ± 0.00350 |
| A | zenith | 93° | 16 | 0.987 / 0.999 / 1.006 | 0.9973 ± 0.0012 | +0.00017 ± 0.00019 | −0.00038 ± 0.00026 |
| A | zenith | 96° | 32 | 0.961 / 0.996 / 1.031 | 0.9916 ± 0.0056 | +0.00065 ± 0.00044 | −0.00086 ± 0.00104 |
| A | zenith | 99° | 16 | 0.684 / 0.971 / 1.203 | 0.9546 ± 0.0220 | −0.00178 ± 0.00274 | +0.00047 ± 0.00609 |
| A | anti-solar | 93° | 16 | 0.967 / 0.976 / 0.995 | 0.9771 ± 0.0010 | −0.00049 ± 0.00009 | −0.00108 ± 0.00017 |
| A | anti-solar | 96° | 16 | 0.980 / 1.004 / 1.038 | 1.0138 ± 0.0052 | −0.00020 ± 0.00048 | +0.00224 ± 0.00089 |
| B | sun side | 94° | 16 | 0.990 / 1.000 / 1.008 | 1.0021 ± 0.0017 | −0.00024 ± 0.00015 | +0.00047 ± 0.00021 |
| B | sun side | 96° | 16 | 0.966 / 0.999 / 1.023 | 1.0018 ± 0.0046 | −0.00118 ± 0.00051 | +0.00051 ± 0.00083 |
| B | anti-solar | 94° | 16 | 0.970 / 0.991 / 1.004 | 0.9945 ± 0.0016 | −0.00014 ± 0.00020 | −0.00022 ± 0.00025 |
| B | anti-solar | 96° | 16 | 0.964 / 1.007 / 1.057 | 0.9955 ± 0.0040 | +0.00094 ± 0.00046 | −0.00187 ± 0.00087 |

Table 5. Absolute same-input comparison of the 1.1.0 engine without NO₂ and O₄ (variant 00).

On the sun side, where the two directions coincide, the luminance ratios lie between 0.9987 and 1.0049. In state A at the
anti-solar pixel at SZA 93 the ratio is 0.977 ± 0.001, 24 standard errors from 1. Four further MYSTIC seeds at the engine's
pixel direction gave 0.995 ± 0.002, against 0.980 ± 0.001 with the same four seeds at the original direction, so most of that
difference comes from the 0.35° offset in elevation. The colour difference at the engine's direction is Δu′ = +0.0001 ±
0.0002 and Δv′ = −0.0002 ± 0.0003. The other zenith and anti-solar pixels were not rerun at the engine's direction.

## Size of the 1.1.0 changes on the production grid

We solved the full-grid gate state on the production grid, in the default band mode, with each combination of the new
absorbers, and compared it with its 1.0.0 table. The state has aod550 0.4, Ångström exponent 1.0, single-scattering albedo
0.92, asymmetry parameter 0.65, a 1.2 km scale height, 300 DU of ozone, a black surface and a stratospheric sulfate layer
with τ550 0.01 at 19 km, with the `usstd76` density and ozone profiles. Y RMS is the root-mean-square relative change of the
luminance over the table pixels whose Y is at least 1 % of the mean of their solar-elevation row, weighted with the cosine of
the view elevation, over all solar-elevation rows of the table. The worst row is the largest RMS of a single solar elevation.
Δu′v′ is the change of chromaticity at each table pixel.

| change against 1.0.0 | Y RMS | worst row | Δu′v′ 95th percentile | Δu′v′ maximum |
|---|---:|---|---:|---:|
| Bodhaine Rayleigh only | 0.45 % | 0.85 % (−4°) | 7.0e-4 | 1.55e-3 |
| and NO₂ (0.207 DU) | 2.40 % | 3.97 % (−18°) | 1.25e-2 | 1.35e-2 |
| and O₄ | 1.67 % | 2.07 % (−12°) | 2.75e-3 | 3.05e-3 |
| and both (the 1.1.0 default) | 4.04 % | 6.18 % (−18°) | 1.08e-2 | 1.17e-2 |

Table 6. Change of the full-grid gate state against its 1.0.0 table.

NO₂ dominates the change of colour, and O₄ offsets part of it. These are differences between two models of one state rather
than errors, and tables from 1.0.0 and 1.1.0 are not interchangeable.

These tables were made before the 1.1.0 change of the interpolation rule of the transport. We solved the 1.1.0 default of
this state, with both absorbers, with the rule before and after the change: the radiance moved by at most 3.7e-6 relative in
every solar-elevation row from +6° to −18°, counting the elements of a row that are at least 1e-3 of the row's maximum, and
its Δu′v′ against 1.0.0 stays 1.08e-2 at the 95th percentile and 1.17e-2 at the maximum. The other three rows of Table 6 were
not re-solved.

## Sensitivity to fixed choices in 1.1.0

We tested two fixed choices of the 1.1.0 model at the tier-1 pixels, against the same tolerance of Δu′v′ 2.9e-4 and 0.122 %
in Y.

**Temperature of the cross sections.** The engine uses NO₂ cross sections at 220 K at all heights and, in the point set-up
above, ozone at 233 K. The default band mode uses an ozone cross section averaged over the profile with the temperatures of
the US Standard Atmosphere (`docs/MODEL.md`, *Ozone*), and that mode was not part of this test. We ran MYSTIC with each
layer's cross sections at the temperature of the US Standard Atmosphere at the layer midpoint and compared with the fixed
temperatures. NO₂ was interpolated between the 220 K and 294 K data of Vandaele et al. (1998), ozone between the Serdyuchenko
columns, and temperatures outside a table were held at its end. The runs used 16 paired seeds at 10 pixels, the tier-1 pixels
without SZA 99. The per-layer temperatures change u′ by −2.3e-5 to −4.8e-5, v′ by −5.1e-5 to −8.0e-5 and Y by −0.004 % to
−0.026 %. The 95 % upper bounds are 1.8e-4 for |Δu′v′|, 0.61 of the tolerance, and 0.040 % for |ΔY|, 0.33 of the tolerance,
so at these pixels the fixed temperatures stay within the tolerance. O₄ kept its fixed temperatures in both runs. Above 666.6
nm the NO₂ asset has no 294 K data, so NO₂ stayed at 220 K there; an estimate of that omitted part, which is not a bound,
gives at most 0.028 % in Y.

**Rayleigh depolarisation.** The phase function uses a depolarisation factor of 0.03 at every wavelength, whereas the King
factor F(λ) of Bodhaine et al. corresponds to ρ = 6(F − 1)/(7F + 3), from 0.0305 at 360 nm to 0.0276 at 830 nm. Paired MYSTIC
runs could not resolve this effect, because a change of the phase function changes the photon paths and the pairs
decorrelate; a pilot at two pixels was dominated by noise, and extrapolating it gives about 25,000 seeds for a result.
Instead, we multiplied each engine pixel spectrum by the ratio of the Rayleigh phase function with ρ(λ) to that with 0.03, at
single scattering angles of 0°, 60°, 90°, 120° and 180°, for the 12 tier-1 pixels. The largest colour change is 3.9e-5 (0.14
of the tolerance) and the largest change in Y is 0.083 % (0.68). This treats all of the radiance as Rayleigh single
scattering at one angle, so it estimates the size of the effect but is not a bound under multiple scattering, which we did
not measure.

## Open checks

- A comparison of the engine's colours with measured twilight colours (for example Lee and Hernández-Andrés 2003). The NO₂
  column of a measured sky is rarely known and varies by a factor of several, and at the pixels above NO₂ moves v′ by
  0.0035–0.0092. The 1.1.0 default is one profile with 0.207 DU, so such a comparison has to treat the column as an unknown
  or take it from a measurement.
- A quantitative comparison of the polarisation with measurements: the absolute DoLP, the depth of the dip, and the inversion
  of the stratospheric optical depth. The Stokes vector comes from MYSTIC on the engine's input, so such a comparison tests
  the input rather than the engine.
- A same-input comparison in the default band mode. Both comparisons with MYSTIC used single wavelengths; the 1.1.0
  comparison used the installed package in point mode at two states and 12 pixels, with SZA up to 99°.
- The wavelength dependence of the Rayleigh depolarisation under multiple scattering. Only the single-scattering estimate
  above exists.
- Temperature dependence of the O₄ cross sections, which the model does not have, and of NO₂ above 666.6 nm, where the asset
  has only 220 K data. Neither was measured.
- Atmospheric refraction. Neither code models it, and MYSTIC's refraction option could not be combined with a reflecting
  surface in this build, so its effect was not measured.

## Reproducing the figure and the records

Figure 1 is rebuilt from the CSV with

```sh
uv run --no-project --with matplotlib --with numpy python docs/validation/plot_validation.py
```

The script reads `docs/validation/mystic_compare.csv` and writes `docs/validation/mystic_vs_engine.png` next to it. The PNG
carries no date or software version, so repeated runs with one matplotlib version give the same bytes. The header of the CSV
defines the columns and the ratio, lists the MYSTIC settings and gives the sha256 of the source records.

`mystic_compare.csv` is the record of the MYSTIC runs and cannot be regenerated from this repository. Regenerating it needs
libRadtran 2.0.6 built with MYSTIC and research scripts that are not distributed here: the writers of the per-layer input
files and of the aerosol species library, the job runner and the comparison script. The engine column needs the table baked
with the research build on a CUDA machine. The full set of MYSTIC runs took 23,897 CPU-seconds of `uvspec` time. The Korkin
check has the same requirements.

`docs/validation/b2_tier1_effects.csv` is the record of the 1.1.0 comparison, with one row for each pixel, variant and
quantity and the columns defined in its header. It cannot be regenerated from this repository either: that needs the same
libRadtran build, a CUDA machine for the engine tables in point mode, and research scripts that are not distributed here,
namely the writer of the per-layer MYSTIC input from the engine's optical input, the paired-seed runner and the comparison
script. The comparison took 35.2 CPU-hours of `uvspec` time, the temperature test 9.6 CPU-hours including its pilot, and the
rerun at the engine's anti-solar pixel direction 0.44 CPU-hours. The absolute 00 table, the phase-function estimate and the
production-grid table are summaries of research records and have no file here.
