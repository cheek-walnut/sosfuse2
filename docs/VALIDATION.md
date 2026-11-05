# Validation

The checks below were run in the research repository that this package was split from, for one atmospheric state. The radiance comparison with MYSTIC uses one sky pixel: the sun side of the sky at 20.25° elevation and 0.5° relative azimuth, seen from 0.1 km above the ground, at solar zenith angles (SZA) of 93°, 96° and 99° (solar elevations −3°, −6° and −9°). The state has a tropospheric aerosol with aod550 0.0551, Ångström exponent 1.889, single-scattering albedo 0.898, Henyey–Greenstein asymmetry parameter 0.683 and a 1.0 km scale height; an ozone column of 224.75 DU; a Gaussian stratospheric sulfate layer (centre 19 km, σ 3 km, τ550 0.0037); and a Lambert surface of albedo 0.376. The engine's output has not been compared with a measured sky.

![Engine and MYSTIC radiance spectra and their ratio at SZA 93, 96 and 99 degrees](validation/mystic_vs_engine.png)

Figure 1. Top: radiance divided by the solar irradiance, L/E₀, from the engine (lines) and from MYSTIC (open markers). The relative standard error of MYSTIC is between 0.52 % and 2.41 % and is not drawn. Bottom: MYSTIC divided by the engine, with the MYSTIC standard error as bars; the three SZA are offset by ±3 nm so that the bars do not overlap. Data: `docs/validation/mystic_compare.csv`. Script: `docs/validation/plot_validation.py`.

## Same-input comparison with MYSTIC

The reference is MYSTIC, the Monte Carlo solver of libRadtran 2.0.6 (Mayer 2009; Emde et al. 2016), run in a 1D spherical-shell atmosphere with polarisation and backward tracing. Both codes were given the same optical input, so a difference between them comes from the transport and its numerics, not from the atmosphere.

The input is the engine's, written out for MYSTIC in 0.5 km layers up to 80 km plus the observer level at 0.1 km: Rayleigh and ozone optical depth per layer, the tropospheric and stratospheric aerosol with their phase functions, and the surface albedo. The Earth radius is 6371 km and the Rayleigh depolarisation factor is 0.03 in both codes. Aerosol scattering is fully depolarising in both: the engine treats it so, and MYSTIC does when it is given only the P11 element of the phase matrix. Neither code models refraction. The MYSTIC input is written by a script that rebuilds the engine's coefficients from the same asset files and state parameters. It does not call the engine, so a mistake in that script would appear as a transport difference. Reading the generated files back through a DISORT direct-beam calculation reproduces the aerosol optical depth to a relative error of at most 5.8e-7 and the total vertical optical depth to at most 3.7e-7, at 430.5, 555.8 and 696.8 nm.

The engine radiance comes from a table baked on 2026-10-27 with the research build. That table uses the legacy spectral inputs: the 48-band solar and ozone table `src/sosfuse2/assets/spectra.json` (the shipped file has the sha256 recorded for the comparison), read as point values at 30 wavelengths, the centres of 30 equal bins over 360–830 nm. MYSTIC ran at the same 30 wavelengths as single wavelengths. The 1.0.0 default on the production grid is different: band means and a temperature-dependent ozone cross section (`src/sosfuse2/spectral.py`). The ozone cross section of the legacy table has a half-bin shift that the research repository corrected on 2026-10-28. Both codes read the same legacy table, so the comparison is not affected by that shift, but it does not test the current spectral defaults. The table was not re-baked with the installed package for this comparison.

Each of the 90 cells (3 SZA × 30 wavelengths) combines two or four independent MYSTIC runs with different random seeds: 260 runs and 2,074,640,958 photons in total. The first pass was too noisy at SZA 99 and in 10 of 30 cells at SZA 96, so those cells received further runs. The cell value is the mean over its runs weighted by photon count, and its standard error (SE) is the photon-weighted combination of the standard errors that MYSTIC reports for each run (`mc_std`). Each SZA has a target for the relative standard error of every cell: 1 % at SZA 93 and 96, 2.5 % at SZA 99.

The ratio below is MYSTIC divided by the engine. Blue is the mean ratio over the wavelengths below 450 nm and red the mean over those above 600 nm. The last column counts the cells whose MYSTIC standard error meets the target. It measures how precise the MYSTIC value is, not how well the two codes agree.

| SZA | ratio min / median / max | blue / red mean ratio | MYSTIC relative SE | cells within target |
|---:|---|---|---|---|
| 93° | 0.9844 / 0.9979 / 1.0102 | 0.9971 / 0.9999 | 0.52–0.77 % | 30 of 30 (1 %) |
| 96° | 0.9890 / 1.0016 / 1.0217 | 1.0017 / 1.0003 | 0.68–0.97 % | 30 of 30 (1 %) |
| 99° | 0.9584 / 1.0040 / 1.0634 | 1.0027 / 1.0105 | 1.97–2.41 % | 30 of 30 (2.5 %) |

The colour is compared through the CIE 1976 coordinates u′ and v′ of the 30-wavelength spectrum, weighted with the solar irradiance and the CIE 1931 colour-matching functions. Δ is MYSTIC minus engine. The standard error comes from 200 random draws of the MYSTIC spectrum within its per-wavelength standard errors, with the engine spectrum held fixed. The draws treat wavelengths as independent, which holds because every cell has its own random seeds.

| SZA | engine u′, v′ | MYSTIC u′, v′ | Δu′ ± SE | Δv′ ± SE |
|---:|---|---|---|---|
| 93° | 0.203054, 0.466584 | 0.203050, 0.466542 | −0.000004 ± 0.000181 | −0.000042 ± 0.000252 |
| 96° | 0.189567, 0.435841 | 0.189052, 0.435705 | −0.000514 ± 0.000316 | −0.000135 ± 0.000496 |
| 99° | 0.191249, 0.432994 | 0.192429, 0.433041 | +0.001181 ± 0.000754 | +0.000047 ± 0.001215 |

At SZA 93 and 96 the ratio of every cell lies within 2.2 % of 1, with a standard error below 1 % per cell. At SZA 99 the ratios range over 0.958–1.063 with a standard error of 2.0–2.4 % per cell. The largest colour differences are Δu′ = −0.0005 ± 0.0003 at SZA 96 and +0.0012 ± 0.0008 at SZA 99, both 1.6 times their standard error. The sign of Δu′ is −, − and + over the three angles, and Δv′ is within one standard error at all three. At the stated precision no systematic difference is detected. For SZA 99, the largest cell deviation |ratio − 1| / SE is 2.69. One of the 30 cells exceeds 2 and none exceeds 3; for 30 independent normal deviates 1.4 cells above 2 are expected.

SZA 99 was run to a relative standard error of 2.5 % per cell instead of the 2 % first specified. The top-up needed for 2 % was planned at 6.05 CPU-hours, above the 6 CPU-hour limit of the specification, so the target was relaxed to 2.5 % (planned 3.64 CPU-hours). The first pass reached 2 % in none of the 30 cells (relative standard error 3.9–18.4 %). After the top-up, 2 of 30 cells meet 2 % and 30 of 30 meet 2.5 %.

The standard errors are Monte Carlo errors of MYSTIC only. They contain no uncertainty of the engine, of the layer discretisation or of the input. Because both codes read the same input, an error in the input itself (the ozone cross section, the Mie table, the solar spectrum, the colour-matching functions, the Rayleigh optical-depth law) is common to both and cancels in the ratio. The comparison does not detect such errors. It covers one state, one pixel and three SZA. Other directions, other states and solar depressions beyond 9° were not run.

## MYSTIC build check

The comparison depends on the MYSTIC build and its spherical 1D set-up. To check them, the same libRadtran build was run on the Rayleigh benchmark of Korkin et al. (2020): a spherical-shell atmosphere with conservative Rayleigh scattering of optical depth 0.25 and 1, a black surface, and the radiance seen at the top of the atmosphere. The paper tabulates MYSTIC results for it. 120 of those cells were reproduced: optical depth 0.25 and 1, solar zenith angles 0°, 70.47° and 84.26°, relative azimuth 0° and 180°, single-scattering and total radiance, 30 cells for each combination of optical depth and mode. Each cell used 1,690,000 photons and one random seed. The `uvspec` binary has sha256 `d0c5ff763780e13455a95f4eaaa7c5ea936d9f1cbdf1d3eeb4c2a9439b42abb8`, which is also the hash of the binary of the tree that ran the comparison above (checked on 2026-11-04).

The largest absolute deviation from the tabulated MYSTIC value is 0.2728 %, and the mean absolute deviation is 0.0643 %. No cell deviates by more than 3 standard errors (0 of 120). The standard error combines the error of the run with the Monte Carlo accuracy that the paper states for each group of cells, because the paper does not give errors per cell. The benchmark values are not reproduced here.

Three limits apply. The paper's vertical structure was not available as a table, so two candidates were tried on three test cells: a uniform shell from 0 to 100 km, and an exponential profile with an 8 km scale height. Only the uniform shell reproduced the table. The exponential profile was 4.2 % and 5.7 % off at two of the cells and 99.99 % low at the third, so the structure was chosen by matching the table. The benchmark neglects polarisation and has no aerosol, no absorption and no reflecting surface, so this check does not test MYSTIC's polarised mode or any of the optical input of the comparison above. The geometry is a top-of-atmosphere view, not a sky seen from the ground.

## Stratospheric aerosol input and zenith polarisation (Ugolnikov et al.)

This calculation was made with MYSTIC, not with the engine. The engine's optical input for the state above, including its stratospheric sulfate layer, was given to MYSTIC, which returns the Stokes vector. The engine has no Stokes output: its tables hold a single radiance per direction and wavelength. The result is therefore a qualitative check of the input, the stratospheric sulfate layer, and does not test the engine's transport.

The calculation looks straight up at SZA 90–99 and 30 wavelengths and takes the degree of linear polarisation, DoLP = √(Q² + U²)/I. Additional runs at SZA 93–96 and 5 wavelengths reduced the standard error where the features below appear. The standard error of a DoLP value is 0.001–0.006 at SZA 91–94, 0.006–0.011 at SZA 95 and 0.011–0.022 at SZA 96.

The reference is published measurements of the polarisation of the zenith twilight sky in a V band (550 nm) and an R band (700 nm) (Ugolnikov et al. 2004). In the V band the measured polarisation is nearly constant until the solar elevation reaches about −5° and then decreases. In the R band the measurements show day-to-day differences in polarisation in bright twilight down to about −3°, after which the differences nearly vanish, and a fall after −5° from multiple scattering. Later papers by Ugolnikov and Maslov relate twilight-sky signals to stratospheric aerosol: a scattering effect that grows with wavelength up to 35 km (Ugolnikov and Maslov 2018a) and a depolarisation near 20 km (Ugolnikov and Maslov 2018b).

Two features are reproduced. The first is a decrease of the polarisation after SZA 95. At 555.8 nm the model DoLP is 0.71–0.73 from SZA 90 to 95 and 0.66 ± 0.01 at SZA 96. The values at SZA 97 and beyond are too noisy to read (0.60 ± 0.06 at SZA 97). The second is an aerosol-caused reduction of the R-band polarisation near the solar elevations where the measurements show day-to-day differences. In the model it appears as a dip with a recovery. At 696.8 nm the DoLP is 0.776 at SZA 91, 0.725 at SZA 93, 0.712 ± 0.003 at SZA 94, 0.755 ± 0.007 at SZA 95 and 0.695 at SZA 96. At 743.8 nm the dip is deeper: 0.782 at SZA 91, 0.700 ± 0.003 at SZA 94 and 0.769 ± 0.007 at SZA 95. The measurements describe a reduction that depends on the day, not a dip, so what is reproduced is the presence of the reduction in this range of solar elevation, not its shape.

To find the cause, the stratospheric aerosol was removed (its column set to zero, everything else equal) and the calculation repeated for SZA 91–96 at 555.8, 696.8 and 743.8 nm. Without the layer the dip and the recovery are gone. At 743.8 nm the DoLP is then 0.840, 0.861, 0.859, 0.849, 0.820 and 0.774 for SZA 91–96: a maximum near SZA 92–93 followed by a steady decrease. The depolarisation by the layer, DoLP with the layer minus DoLP without it, is −0.058, −0.086, −0.132, −0.149 and −0.051 at 743.8 nm for SZA 91–95. At 555.8 nm it is at most −0.073 (SZA 93), about half the largest value at 743.8 nm. The direction agrees with the measurements, in which the aerosol effect is larger at longer wavelengths.

No quantitative comparison of the absolute polarisation or of the depth of the dip was made. The published values exist only as figures, and the observing sites, dates and aerosol loads differ from the state used here. In this calculation the aerosol scatters with P11 only, so it depolarises completely, as in the engine. A full Mie phase matrix was not used, and the effect of that assumption on the depth of the dip was not tested. The research repository names a next step that has not been taken: to invert the stratospheric optical depth from measured polarisation with the relation of Ugolnikov and Maslov (2019) and to compare it with the input value of 0.0037.

## Input physics not in the default model

Three differences between the engine's default input and libRadtran's own molecular input were measured by running MYSTIC twice, with and without the item, on otherwise identical input. They are choices of input physics in the default model, not errors of the solver. The engine has no NO2 and no O4 absorption, and it scales the Rayleigh optical depth as a pure power law in wavelength. The same-input comparison above cannot detect any of them. The numbers below are differences between two MYSTIC runs, not between MYSTIC and the engine. For scale, the research repository judges changes of the solver against a colour budget of Δu′v′ 2.9e-4 at the 95th percentile.

### NO2

libRadtran's `afglus` profile contains NO2 with a column of 5.57e15 cm⁻² (0.207 DU), a vertical optical depth of about 3e-3 between 370 and 430 nm. To measure its effect, two MYSTIC runs that differ only in NO2 were given the same random seeds. Their photon paths are then nearly identical and the ratio is precise. Each pixel below has 8 seeds of 500,000 photons at each of 30 wavelengths. The ratio is with NO2 divided by without, and Δ is with minus without. The engine corresponds to the run without NO2. ΔY is the relative change of the luminance Y, with the same weights as the colour above.

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

In all eight pixels NO2 lowers the blue radiance and raises v′. An engine without NO2 therefore gives a lower v′, a bluer sky, than the same sky with this NO2 column, by 0.0035 to 0.0092. The change is 12 to 32 times the colour budget quoted above. It increases with SZA and varies by about ±25 % with direction at a given SZA. At the zenith at SZA 99 the blue is so dark that rare high-order scattering paths dominate and the two runs decorrelate, which is why the errors are large there, but Δv′ is still 7 times its standard error. Two independent runs (different seeds) at the sun-side pixel at SZA 96 give Δv′ = +0.0049 ± 0.0006, consistent with the paired value of +0.00480 ± 0.00006.

The column is that of one profile. Stratospheric NO2 varies by a factor of several with season, latitude and twilight chemistry, so the size of the effect for a given sky is not known from these runs. It is expected to scale roughly with the column; this was not measured.

### O4

libRadtran includes the O4 absorption computed from the O2 profile, with a vertical optical depth of 8.3e-3 at 477.5 nm, its maximum, and 7.4e-3 at 571.5 and 634.2 nm. At the sun-side pixel at SZA 96, adding O4 to MYSTIC changes the radiance by factors of 0.966, 0.939, 0.964 and 0.913 at 477.5, 571.5, 587.2 and 634.2 nm. The factor is about 1 at the other wavelengths. The colour changes by Δu′ = −0.0013 ± 0.0004 and Δv′ = −0.0007 ± 0.0006. These are two independent runs with two seeds of 9 million photons per wavelength. The 30 wavelengths are point samples, so the narrow O4 bands are sampled at their centres and the effect on band-mean spectra is expected to be smaller. That was not measured.

### Rayleigh optical-depth law

The engine takes the Rayleigh optical depth at 550 nm from the IPRT 450 nm table for the US Standard Atmosphere multiplied by (450/550)⁴ (`src/sosfuse2/generate_density_profile.py`, line 89), and scales it with wavelength as (550/λ)⁴ (`src/sosfuse2/accepted/sos_sky.py`, line 40). The Rayleigh optical depth of Bodhaine et al. (1999), which the libRadtran run used (`crs_model rayleigh Bodhaine`), is not an exact λ⁻⁴ law. For libRadtran's `afglus` atmosphere the ratio of the Bodhaine optical depth to the engine's is 1.031 at 368 nm, 1.000 at 446 nm and 0.957 at 822 nm, measured as vertical optical depth in a DISORT direct-beam calculation.

Scaling the engine's Rayleigh optical depth wavelength by wavelength to the Bodhaine value changes the MYSTIC radiance at the sun-side pixel at SZA 96 by an inverse-variance weighted mean ratio of 0.9892 ± 0.0020, 5.4 standard errors from 1. The ratio is 0.9825 ± 0.0033 below 420 nm and above 700 nm, and 0.9917 ± 0.0030 between 450 and 650 nm. The colour does not move beyond its standard error: Δu′ = +0.0001 ± 0.0004 and Δv′ = +0.0001 ± 0.0006. Only the optical depth was changed, and the depolarisation factor of 0.03 was kept.

### Scope

These three effects were measured for one state. The pair runs for O4 and Rayleigh are at SZA 96, the NO2 runs at SZA 93–99, all in the directions listed. Band-mean spectra, other states and other directions were not measured. Adding NO2 as an absorber or replacing the Rayleigh law would change the default model and the numerical references that depend on it. Neither has been done.

## Not yet done

- A comparison of the engine's colours with measured twilight colours (for example Lee and Hernández-Andrés 2003). Such a comparison has to carry the NO2 sensitivity above or use an engine with NO2. The research record notes that a gap of 0.004–0.006 in v′ between model and measurement could come from the missing NO2 alone.
- A quantitative comparison of the polarisation with measurements: the absolute DoLP, the depth of the dip, and the inversion of the stratospheric optical depth. The Stokes vector comes from MYSTIC on the engine's input, so such a comparison tests the input, not the engine.
- The same-input comparison with tables baked by the installed package and with the 1.0.0 spectral defaults, and for more than one state, pixel and SZA range.
- Atmospheric refraction. Neither code models it, and MYSTIC's refraction option could not be combined with a reflecting surface in this build, so its effect was not measured.

## Reproducing

Figure 1 is rebuilt from the CSV with

```sh
uv run --no-project --with matplotlib --with numpy python docs/validation/plot_validation.py
```

The script reads `docs/validation/mystic_compare.csv` and writes `docs/validation/mystic_vs_engine.png` next to it. The PNG carries no date or software version, so repeated runs with one matplotlib version give the same bytes. The header of the CSV defines the columns and the ratio, lists the MYSTIC settings and gives the sha256 of the source records.

The CSV is the record of the MYSTIC runs. It cannot be regenerated from this repository. That needs libRadtran 2.0.6 built with MYSTIC and research scripts that are not in this repository: the writers of the per-layer input files and of the aerosol species library, the job runner and the comparison script. The engine column needs the table baked with the research build on a CUDA machine. The full set of MYSTIC runs took 23,897 CPU-seconds of `uvspec` time. The Korkin check has the same requirements.
