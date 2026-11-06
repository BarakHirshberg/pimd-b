# Ladder regression checks

Build in the pinned benchmark environment. On HPC, run all compilation and tests
on allocated compute nodes, never the login node.

Configure CMake with `-DNDIM=3 -DBUILD_LADDER_TESTS=ON`, build, and run
`ctest --test-dir <build> --output-on-failure`. Then set `OPENMPI_BIN` to the
environment's bin directory and run `python tests/main.py <absolute-build-path>
<absolute-tests/cases-path>` for the saved-output regressions.

Saved-output fixtures are three-dimensional. Repeat CTest with NDIM=1 and NDIM=2
for dimensional force coverage. CTest needs four ranks on one node; regression
fixtures need up to eight. CI enables oversubscription on its small runner.

CTest checks the recursion against brute-force compositions, probabilities and
conditional permutation measurements, then MPI force derivatives with and
without periodic image sums. Softening in the force test is a fixed parameter,
not a Monte Carlo sampling move. No test simulates path updates by MC.

The regression runner selects complete named fixtures (case.ini and
simulation.out), excluding the other compile-time algorithm. Helper fixtures
are not regression trajectories. It fails if no regression cases are found.

The two-rank `exchange_benchmark config.ini` times fixed-coordinate exchange
prepare/force/primitive-estimator evaluation and conditional permutation draws
separately. Process RSS includes both exchange objects; it does not measure the
ladder object's memory scaling separately.

Scientific validation lives in hebench's ideal_validation module. Compare
physical averages with exact references: ladder and quadratic potentials and
forces intentionally differ at fixed coordinates.

The runner also rejects missing observable columns and array-shape differences;
NumPy broadcasting cannot turn missing trajectory frames into a passing result.
Run `python -m unittest discover -s tests -p test_regression_runner.py` to check
those failure cases and compatibility with the saved fixtures. CI runs it too.
