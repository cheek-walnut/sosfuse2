# Third-party notices

This file lists every file in the distributed package (sdist and wheel) that is derived from a third-party source,
with the original notice, the licence text, what the file is used for and how it was changed. Installation
dependencies (NumPy, Pillow, CuPy, pytest, the NVIDIA CUDA wheels) are not covered here; their licences come with the
packages themselves. Everything not listed below is this project's own work: code under the MIT licence (`LICENSE`),
generated data under CC0 1.0 (`src/sosfuse2/assets/README.md`).

Notices attach to the **inputs** of a generator, not to the tables generated from them: a Mie table computed by this
project from published aerosol parameters is project data; a solar spectrum resampled from a published spectrum keeps
the publisher's attribution.

## 1. Mitsuba 3 — CIE 1931 colour matching functions

- Files: `src/sosfuse2/assets/cie1931_5nm_local.json` (CIE 1931 2° XYZ, 360–830 nm at 5 nm, 95 samples, read from the
  table compiled into Mitsuba 3.9.1 `libmitsuba`; the file records the library and header SHA-256 and the byte offset),
  `src/sosfuse2/assets/cmf_1nm.json` (CIE 1931 XYZ at 1 nm, `mitsuba.cie1931_xyz` of Mitsuba 3.9.1 `scalar_rgb`).
- Use: colour weights of the 15 spectral bins (`spectral.py`, `rgb_weights.npy`), colour figures in the research tools.
- Changes: numeric values unchanged; stored as JSON with provenance fields. The original source locations were replaced by
  package-relative descriptors before publication.
- The CIE 1931 standard observer itself is a CIE standard data set; the table here is the copy shipped by Mitsuba.
- Originator attribution: the numbers are the *CIE 1931 colour-matching functions, 2° observer*. CIE distributes this data set
  itself (CIE Publications-datasets, https://cie.co.at/data-tables) under **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/);
  attributed here as © CIE, CC BY-SA 4.0.
  This project took the copy compiled into Mitsuba, not CIE's file; whether CIE's licence of its own download applies to a copy
  obtained through a BSD-licensed library was not established, so the distribution's licence expression (`pyproject.toml`)
  names both BSD-3-Clause and CC-BY-SA-4.0 (2026-11-03 release review).
- Licence (Mitsuba 3, https://github.com/mitsuba-renderer/mitsuba3/blob/master/LICENSE):

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

- File: `src/sosfuse2/assets/lef_cie.json` (CIE 1924 photopic V(λ) and CIE 1951 scotopic V′(λ), 380–780 nm, with
  K_m constants; values from colour-science 0.4.7 `SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"]` and
  `SDS_LEFS_SCOTOPIC["CIE 1951 Scotopic Standard Observer"]`).
- Use: the scotopic V′(λ) weights the S′ channel of the colour weights (`spectral.channel_weights`). Luminance Y comes from
  the CIE 1931 ȳ; the photopic V(λ) column is stored but not read.
- Changes: numeric values unchanged; stored as JSON, zero outside 380–780 nm, mesopic cross-check values appended.
- Originator attribution: CIE 1924 photopic V(λ) and CIE 1951 scotopic V′(λ) are CIE standard data sets, distributed by CIE
  under CC BY-SA 4.0 (same note as §1); attributed here as © CIE, CC BY-SA 4.0. The copy came from colour-science (BSD-3-Clause below).
- Licence (colour-science, https://github.com/colour-science/colour/blob/develop/LICENSE):

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

- File: `src/sosfuse2/assets/spectra.json` (solar irradiance and ozone cross sections on the 48 wavelengths of
  `atmosphere/demo/demo.cc`; the file records the source URL and the SHA-256 of the fetched source).
- Use: spectral inputs of the frozen accepted solver sources (`accepted/`), read by the legacy spectral mode and by the
  reduced single-wavelength path. The default full-grid tables use the TSIS-1 HSRS and IUP Bremen assets (§4, §5).
- Changes: numbers transcribed from the C++ source into JSON with units; the demo's own documentation attributes the
  solar values to ASTM G173 and the ozone values to the Bremen cross-section data base.
- Licence (https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/LICENSE):

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

- Files: the `solar` arrays of `src/sosfuse2/assets/spectra_fine.json` and `spectra_fine_temperature.json`
  (0.1 nm cell means of the full-resolution HSRS, 360–840 nm, air wavelengths; the `sources.solar` key in each file
  records the source file, its SHA-256 and the scaling applied).
- Source: LASP LISIRD, *TSIS-1 HSRS Version 2* (`hybrid_reference_spectrum_c2022-11-30_with_unc.nc`),
  https://lasp.colorado.edu/lisird/data/tsis1_hsrs . The data are distributed openly by LASP; cite the references below
  when using them.
- Redistribution: LASP publishes the HSRS files for open download and asks for citation; no licence text accompanies the file,
  and redistribution of a resampled copy was not confirmed with LASP. What ships here is the 0.1 nm cell means, not the original file.
- References: Coddington, O. M., et al. (2021). *The TSIS-1 Hybrid Solar Reference Spectrum.* Geophysical Research
  Letters, 48, e2020GL091709. Coddington, O. M., et al. (2023). *Version 2 of the TSIS-1 Hybrid Solar Reference Spectrum
  and Extension to the Full Spectrum.* Earth and Space Science, 10, e2022EA002637.
- Changes: averaged onto a 0.1 nm grid, converted to air wavelengths (Ciddor 1996), scaled to the TSIS-1 TIM total solar
  irradiance as described in the file.

## 5. Serdyuchenko & Gorshelev ozone absorption cross sections (IUP Bremen)

- Files: the `ozone_*K` arrays of `spectra_fine.json` (213, 233, 253 K) and `spectra_fine_temperature.json`
  (193–293 K); `sources.ozone` records the source file name, URL and SHA-256.
- Source: Institute of Environmental Physics, University of Bremen, molecular spectroscopy group,
  `serdyuchenkogorshelev5digits.dat` (https://www.iup.uni-bremen.de/gruppen/molspec/). Published scientific data;
  cite the references below when using them.
- Redistribution: the IUP Bremen pages publish the data base for download with a request to cite the two papers; no licence text
  accompanies the file, and redistribution of the interpolated copy was not confirmed with IUP Bremen.
- References: Serdyuchenko, A., Gorshelev, V., Weber, M., Chehade, W., Burrows, J. P. (2014). *High spectral resolution
  ozone absorption cross-sections – Part 2: Temperature dependence.* Atmos. Meas. Tech., 7, 625–636.
  Gorshelev, V., Serdyuchenko, A., Weber, M., Chehade, W., Burrows, J. P. (2014). *Part 1: Measurements, data analysis
  and comparison with previous measurements around 293 K.* Atmos. Meas. Tech., 7, 609–624.
- Changes: unit conversion cm² → m², relabelled to air wavelengths, interpolated once onto the 0.1 nm grid.

## 6. LOWTRAN7 — tabulated aerosol numbers

- Where: numeric constants transcribed into `src/sosfuse2/generate_tropo_optics.py` and `src/sosfuse2/tropo_profile.py`;
  in `src/sosfuse2/assets/tropo_rural_sf79_rh70.json` the fields `lowtran7.rows[*].{ext_rel_lowtran, ssa_lowtran, g_lowtran}`
  are LOWTRAN7 table values kept beside this project's `*_mie` values for validation (the rest of that file is project data, CC0)
  (extinction/profile block data of LOWTRAN7, Kneizys et al. 1988, Air Force Geophysics Laboratory), used to validate
  the rural aerosol table and to build the aerosol height profile.
- Terms: LOWTRAN7 was produced by the United States Air Force Geophysics Laboratory; works of the U.S. Government are
  not subject to copyright in the United States. This status is what the project relies on; it has not been confirmed
  with the publisher. The source URL is recorded beside the constants.
- Reference: Kneizys, F. X., et al. (1988). *Users Guide to LOWTRAN 7.* AFGL-TR-88-0177.

## 7. Shettle & Fenn 1979 — aerosol model parameters

- Where: input parameters (mode radii, widths, refractive indices, humidity growth) of `generate_tropo_optics.py`,
  recorded in `tropo_rural_sf79_rh70.json` under `particle`.
- Terms: published parameters from a public technical report; the Mie table computed from them is project data (CC0).
- Reference: Shettle, E. P., Fenn, R. W. (1979). *Models for the Aerosols of the Lower Atmosphere and the Effects of
  Humidity Variations on Their Optical Properties.* AFGL-TR-79-0214.

## 8. U.S. Standard Atmosphere 1976 — formulas

- Where: `generate_density_profile.py` (density and temperature profile formulas), `temperature_usstd76.json`.
- Terms: U.S. Government publication (NOAA/NASA/USAF 1976); not subject to copyright in the United States.

## 9. Not bundled: IPRT ozone profile shape

- `iprt.py` / `generate_density_profile.py` download two files, `tau_rayleigh_450nm_usstd.dat` and
  `tau_absorption_320nm_usstd.dat`, from the phase E page (`intercomparisons:phase_e`) of the IPRT (International Polarized
  Radiative Transfer) intercomparison pages hosted by libRadtran (https://www.libradtran.org/) the first time a state selects a `usstd76` density or ozone
  profile (the default states do not), verify their
  SHA-256 and derive the `usstd76` density asset in the user's work root. The terms of these files were not confirmed
  (the wiki footer states CC BY-SA 3.0 for wiki content without saying whether attachments are covered), so they are not
  redistributed with this package. `SOSFUSE2_OFFLINE=1` refuses the download and prints the URL to fetch by hand.
- Reference: Emde, C., et al. (2015). *IPRT polarized radiative transfer model intercomparison project – Phase A.*
  JQSRT, 164, 8–36. Emde, C., et al. (2018). *IPRT polarized radiative transfer model intercomparison project –
  Three-dimensional test cases (phase B).* JQSRT, 209, 19–44. https://doi.org/10.1016/j.jqsrt.2018.01.024

## 10. Algorithms cited, no code copied

- Bohren, C. F., Huffman, D. R. (1983). *Absorption and Scattering of Light by Small Particles.* The BHMIE recursion
  structure is reimplemented in `mie.py`.
- Ciddor, P. E. (1996). *Refractive index of air: new equations for the visible and near infrared.* Applied Optics 35,
  1566 — `air_wavelength.py`.
- Hable, J. (2010). *Uncharted 2: HDR Lighting* (GDC 2010). Constants of J. Hable's 2010 filmic tone curve in `accepted/sos_render.py` `film()`; formula only, no code copied. Used for the preview PNGs written by `host/pack_state.py`.
