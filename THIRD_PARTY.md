# Third-party notices

This file lists every file in the distributed package, sdist and wheel, that derives from a third-party source, with the
original notice, the licence text, what the file is used for and how it was changed. Installation dependencies (NumPy,
Pillow, CuPy, pytest, the NVIDIA CUDA wheels) are not covered; their licences come with the packages themselves.
Everything not listed below is this project's own work: code under the MIT licence (`LICENSE`) and generated data under
CC0 1.0 (`src/sosfuse2/assets/README.md`).

A notice attaches to the inputs of a generator rather than to the tables generated from them. A Mie table that this
project computed from published aerosol parameters is project data, whereas a solar spectrum resampled from a published
spectrum keeps the publisher's attribution.

Several of the sources below (§4, §5, §6, §9, §12) publish their data for open download without a licence text. We rely
on that open publication and on the publishers' citation requests, and we have not asked any of them to confirm that a
resampled or transcribed copy may be redistributed. Where that uncertainty seemed too large, the data is not bundled and
the user downloads it instead (§9, §12).

## 1. Mitsuba 3 — CIE 1931 colour matching functions

The files `src/sosfuse2/assets/cie1931_5nm_local.json` and `src/sosfuse2/assets/cmf_1nm.json` hold the CIE 1931 2° XYZ
colour-matching functions. The first has 95 samples from 360 to 830 nm at 5 nm, read from the table compiled into
Mitsuba 3.9.1 `libmitsuba`, and records the library and header SHA-256 and the byte offset. The second is the 1 nm table,
`mitsuba.cie1931_xyz` of Mitsuba 3.9.1 `scalar_rgb`. They provide the colour weights of the 15 spectral bins
(`atmosphere/spectral.py`, `rgb_weights.npy`) and the colour figures of the research tools. The numeric values are unchanged; we
stored them as JSON with provenance fields and replaced the original source locations by package-relative descriptors
before publication.

The numbers themselves are the CIE 1931 colour-matching functions for the 2° observer, a CIE standard data set that CIE
distributes (CIE Publications-datasets, https://cie.co.at/data-tables) under **CC BY-SA 4.0**
(https://creativecommons.org/licenses/by-sa/4.0/). We attribute them as © CIE, CC BY-SA 4.0. Our copy came through
Mitsuba, a BSD-licensed library, rather than from CIE's own download, and we could not establish which licence governs a
copy obtained that way, so the distribution's licence expression in `pyproject.toml` names both BSD-3-Clause and
CC-BY-SA-4.0.

Licence of Mitsuba 3 (https://github.com/mitsuba-renderer/mitsuba3/blob/master/LICENSE):

```
Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
```

## 2. colour-science — CIE luminous efficiency functions

`src/sosfuse2/assets/lef_cie.json` holds the CIE 1924 photopic V(λ) and the CIE 1951 scotopic V′(λ) from 380 to 780 nm
with their K_m constants. The values come from colour-science 0.4.7,
`SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"]` and `SDS_LEFS_SCOTOPIC["CIE 1951 Scotopic Standard Observer"]`.
The scotopic V′(λ) weights the S′ channel of the colour weights (`atmosphere/spectral.py::channel_weights`); luminance Y comes from the
CIE 1931 ȳ instead, so the photopic V(λ) column is stored but never read. The numeric values are unchanged. We stored
them as JSON, set them to zero outside 380–780 nm and appended mesopic cross-check values.

Both functions are CIE standard data sets, distributed by CIE under CC BY-SA 4.0 as described in §1, and we attribute
them as © CIE, CC BY-SA 4.0. The copy came from colour-science, whose BSD-3-Clause licence follows.

Licence of colour-science (https://github.com/colour-science/colour/blob/develop/LICENSE):

```
Copyright 2013 Colour Developers

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 3. Eric Bruneton, *Precomputed Atmospheric Scattering* — 48-band spectra

`src/sosfuse2/assets/spectra.json` holds the solar irradiance and ozone cross sections on the 48 wavelengths of
`atmosphere/demo/demo.cc`, and records the source URL and the SHA-256 of the fetched source. The CPU engine
(`solve/engine.py`) reads it as its spectral input, as do the legacy spectral mode and the reduced single-wavelength
path; the default full-grid tables use the TSIS-1 HSRS and IUP Bremen assets of §4 and §5 instead. We transcribed the
numbers from the C++ source into JSON with units. The demo's own documentation attributes the solar values to ASTM G173
and the ozone values to the Bremen cross-section data base.

Licence (https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/LICENSE):

```
Copyright (c) 2017 Eric Bruneton
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## 4. TSIS-1 Hybrid Solar Reference Spectrum (HSRS) v2 — fine solar spectrum

The `solar` arrays of `src/sosfuse2/assets/spectra_fine.json` and `spectra_fine_temperature.json` are 0.1 nm cell means
of the full-resolution HSRS from 360 to 840 nm, on air wavelengths. The `sources.solar` key in each file records the
source file, its SHA-256 and the scaling applied.

The source is LASP LISIRD, *TSIS-1 HSRS Version 2* (`hybrid_reference_spectrum_c2022-11-30_with_unc.nc`),
https://lasp.colorado.edu/lisird/data/tsis1_hsrs . LASP publishes the HSRS files for open download and asks for citation;
no licence text accompanies the file. What ships here is the 0.1 nm cell means, not the original file. We averaged the
spectrum onto a 0.1 nm grid, converted it to air wavelengths (Ciddor 1996) and scaled it to the TSIS-1 TIM total solar
irradiance as described in the file.

References: Coddington, O. M., et al. (2021). *The TSIS-1 Hybrid Solar Reference Spectrum.* Geophysical Research Letters,
48, e2020GL091709. Coddington, O. M., et al. (2023). *Version 2 of the TSIS-1 Hybrid Solar Reference Spectrum and
Extension to the Full Spectrum.* Earth and Space Science, 10, e2022EA002637.

## 5. Serdyuchenko & Gorshelev ozone absorption cross sections (IUP Bremen)

The `ozone_*K` arrays of `spectra_fine.json` (213, 233, 253 K) and `spectra_fine_temperature.json` (193–293 K) come from
`serdyuchenkogorshelev5digits.dat` of the molecular spectroscopy group at the Institute of Environmental Physics,
University of Bremen (https://www.iup.uni-bremen.de/gruppen/molspec/); `sources.ozone` records the source file name, URL
and SHA-256. The IUP Bremen pages publish the data base for download with a request to cite the two papers below; no
licence text accompanies the file. We converted the units from cm² to m², relabelled the wavelengths to air and
interpolated once onto the 0.1 nm grid.

References: Serdyuchenko, A., Gorshelev, V., Weber, M., Chehade, W., Burrows, J. P. (2014). *High spectral resolution
ozone absorption cross-sections – Part 2: Temperature dependence.* Atmos. Meas. Tech., 7, 625–636. Gorshelev, V.,
Serdyuchenko, A., Weber, M., Chehade, W., Burrows, J. P. (2014). *Part 1: Measurements, data analysis and comparison with
previous measurements around 293 K.* Atmos. Meas. Tech., 7, 609–624.

## 6. LOWTRAN7 — tabulated aerosol numbers

Aerosol extinction and profile constants from LOWTRAN 7 (Kneizys et al. 1988, Air Force Geophysics Laboratory) are
transcribed into `src/sosfuse2/asset_tools/generate_tropo_optics.py` and `src/sosfuse2/atmosphere/tropo_profile.py`, where they validate the
rural aerosol table and build the aerosol height profile. In `src/sosfuse2/assets/tropo_rural_sf79_rh70.json` the fields
`lowtran7.rows[*].{ext_rel_lowtran, ssa_lowtran, g_lowtran}` are LOWTRAN7 table values kept beside this project's `*_mie`
values for validation; the rest of that file is project data under CC0. The source URL is recorded beside the constants.

LOWTRAN7 was produced by the United States Air Force Geophysics Laboratory, and works of the U.S. Government are not
subject to copyright in the United States. The project relies on that status.

Reference: Kneizys, F. X., et al. (1988). *Users Guide to LOWTRAN 7.* AFGL-TR-88-0177.

## 7. Shettle & Fenn 1979 — aerosol model parameters

The mode radii, widths, refractive indices and humidity growth that `asset_tools/generate_tropo_optics.py` takes as input are the
published parameters of a public technical report, recorded in `tropo_rural_sf79_rh70.json` under `particle`. The Mie
table computed from them is project data under CC0.

Reference: Shettle, E. P., Fenn, R. W. (1979). *Models for the Aerosols of the Lower Atmosphere and the Effects of
Humidity Variations on Their Optical Properties.* AFGL-TR-79-0214.

## 8. U.S. Standard Atmosphere 1976 — formulas

`asset_tools/generate_density_profile.py` implements the density and temperature profile formulas, and `temperature_usstd76.json`
holds the resulting temperature profile. The standard is a U.S. Government publication (NOAA/NASA/USAF 1976) and is not
subject to copyright in the United States.

## 9. Not bundled: IPRT ozone profile shape

The first time a state uses a density configuration other than `exp8`/`triangle`/`keep`, the default
`usstd76`/`usstd76`/`usstd` included, `asset_tools/iprt.py` and `asset_tools/generate_density_profile.py` download two
files, `tau_rayleigh_450nm_usstd.dat` and `tau_absorption_320nm_usstd.dat`, from the phase E page
(`intercomparisons:phase_e`) of the IPRT (International Polarized Radiative Transfer) intercomparison pages hosted by
libRadtran (https://www.libradtran.org/), verify their SHA-256 and derive the `usstd76` density asset in the user's work
root. Setting all three density options to `exp8`/`triangle`/`keep` avoids the download. The wiki footer states CC BY-SA
3.0 for wiki content without saying whether attachments are covered, so the files are not redistributed with this
package. `SOSFUSE2_OFFLINE=1` refuses the download and prints the URL to fetch by hand.

References: Emde, C., et al. (2015). *IPRT polarized radiative transfer model intercomparison project – Phase A.* JQSRT,
164, 8–36. Emde, C., et al. (2018). *IPRT polarized radiative transfer model intercomparison project – Three-dimensional
test cases (phase B).* JQSRT, 209, 19–44. https://doi.org/10.1016/j.jqsrt.2018.01.024

## 10. Algorithms cited, no code copied

- Bodhaine, B. A., Wood, N. B., Dutton, E. G., Slusser, J. R. (1999). *On Rayleigh Optical Depth Calculations.*
  Journal of Atmospheric and Oceanic Technology 16, 1854–1861. `atmosphere/rayleigh.py` implements their equations 18, 19, 5, 6, 23
  and 22 with 360 ppm CO₂, at vacuum wavelengths obtained from Ciddor standard-air coordinates. The engine keeps its
  original optical depth at 450 nm in air and its phase depolarization of 0.03.
- Bohren, C. F., Huffman, D. R. (1983). *Absorption and Scattering of Light by Small Particles.* `asset_tools/mie.py` reimplements
  the structure of the BHMIE recursion.
- Ciddor, P. E. (1996). *Refractive index of air: new equations for the visible and near infrared.* Applied Optics 35,
  1566. Implemented in `atmosphere/air_wavelength.py`.
- Hable, J. (2010). *Uncharted 2: HDR Lighting* (GDC 2010). `outputs/preview_functions.py` `film()` uses the constants of Hable's
  2010 filmic tone curve for the preview PNGs written by `outputs/pack_state.py`; formula only, no code copied.

## 11. Bundled: Vandaele NO₂ absorption at 220 K

`src/sosfuse2/assets/no2_vandaele1998_fine.json` derives from the BIRA-IASB files `no2c_97.dat` and `no2cp_97.dat`.

Attribution: Robert, S., Vandaele, A. C., Hermans, C. (2026). *NO2 absorption cross sections at different temperatures
and pressures (Version 1).* Royal Belgian Institute for Space Aeronomy. https://doi.org/10.18758/6b80spv8. Measurements:
Vandaele, A. C., et al. (1998). JQSRT 59, 171–184.

Licence: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/. This asset is excluded from the project's CC0
dedication.

Changes: column 1 of the source gives the inverse air wavelength. We use C_97 column 3 below 660 nm, a linear blend over
660–666 nm, and the unscaled CP_97 column 6 (0.16 Torr) above 666 nm; integrate the source's linear interpolant into
0.1 nm cells; convert cm² to m²; and round to seven significant digits. End padding is removed, and internal missing rows
are refused. The asset records the original digests, units, changes and generator version.

## 12. Not bundled: HITRAN2024 O₂–O₂ collision-induced absorption (O₄)

Neither `O2-O2_2024.cia` nor the O₄ JSON derived from it is included in git or in the wheel. The source is
https://hitran.org/data/CIA/main/O2-O2_2024.cia. The user prepares the table with
`python -m sosfuse2.asset_tools.prepare_minor_absorbers --cia FILE`, or with `--download` to retrieve the official source. The
command verifies the pinned SHA-256 of the original, writes `assets/o4_hitran2024_fine.json` atomically under
`SOSFUSE2_WORK_ROOT` (otherwise the XDG work root) and prints the SHA-256 of the result. A solve only reads this cache,
and `SOSFUSE2_OFFLINE=1` refuses downloads. Run the preparation once before precompiling, baking or running a gate that
uses the 1.1.0 absorber fixtures.

References: Gordon, I. E., et al. (2026). *The HITRAN2024 molecular spectroscopic database.* JQSRT 353, 109807; Terragni,
J., et al. (2025). *Collision induced absorption in HITRAN2024.* JQSRT 347, 109631; Thalman, R., Volkamer, R. (2013). PCCP
15, 15371; Finkenzeller, H., Volkamer, R. (2022). JQSRT 279, 108063.

Changes: the generator converts vacuum wavenumber to standard-air wavelength and uses HITRAN reference 54 at 293 K up to
its last point at 499.861 nm, then reference 32 at 293 K up to 600 nm and at 287 K up to 653.842 nm, with zero outside
that range. HITRAN recommends reference 32 only from 509.334 nm, so using it already from 499.861 nm is this project's own
join rule across the 499.861–509.334 nm gap; `asset_tools/generate_minor_absorbers.py` records it in the output under
`rules.connection`. The values are integrated into 0.1 nm cells, negative cell means are clipped to zero, cm⁵ is converted
to m⁵ and the result is rounded to seven significant digits. The O₂ A- and B-band CIA sets are excluded. The local output
records the source SHA-256 and the rules version.

## 13. Bundled: AFGL U.S. Standard Atmosphere NO₂ profile

`src/sosfuse2/assets/no2_afglus_profile.json` derives from `afglus.dat` in AFGL-TR-86-0110, obtained through libRadtran
2.0.6. The source is U.S. Government work. We interpolated the NO₂ number density log-linearly onto a 0.5 km grid. The
profile supplies only the shape; the state sets the NO₂ column. The asset records the source SHA-256 and the
interpolation.
