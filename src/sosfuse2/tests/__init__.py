"""The packaged tests: `pytest --pyargs sosfuse2` collects them from the installed package (tier 1 of docs/CHECKS.md).

Tests that mention `scripts/sosfuse2/...` or `scripts/sosfuse2_dev/...` read the sky-sos research checkout (tier 3,
`SOSFUSE2_SKY_SOS_ROOT`): those are paths inside that checkout or in its git history, not paths of this package, and the
tests skip without it.
"""
