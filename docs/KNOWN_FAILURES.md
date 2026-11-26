# Known failures

As of 1.6.1, the test suite has no cases marked as expected failures.

## Environment-dependent cases

Some cases depend on the host. The kernel syntax tests skip without `clang++` or `clang` on `PATH`. The persistent-queue
tests fail with `TimeoutExpired` when `PERSISTENT_TEST_DEADLINE` is too short for the host, and on macOS they skip with
`E_SOCKET_PATH_TOO_LONG` when `TMPDIR` makes the socket path, terminator included, longer than 104 bytes. See tier 1 of `docs/CHECKS.md`
for the test requirements.
