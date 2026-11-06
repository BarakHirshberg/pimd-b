/**
 * @file exchange_benchmark.cpp
 * @brief Time actual exchange prepare/force/estimator and conditional measurement draws.
 *
 * Fixed configurations only: this is not a trajectory or a sampling algorithm.
 * Launch with two MPI ranks and an input with P=2. Timings exclude initialization,
 * I/O and MPI synchronization; max over the two end ranks is reported. Each round
 * warms the implementation before measuring. Compare dimensions, N and boundary
 * conditions using identical inputs and hardware for the two implementations.
 */
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <sys/resource.h>
#include "mpi.h"
#include "params.h"
#include "simulation.h"
#include "bosonic_exchange/quadratic_bosonic_exchange.h"
#include "bosonic_exchange/ladder_bosonic_exchange.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 2 || ranks != 2) {
        if (rank == 0) std::fprintf(stderr, "usage: mpirun -np 2 exchange_benchmark config.ini\n");
        MPI_Finalize(); return 2;
    }
    {
        Params params(argv[1], rank);
        Simulation sim(rank, ranks, params, 991u);
        sim.updateNeighboringCoordinates();
        BosonicExchange quad(sim);
        LadderBosonicExchange ladder(sim);
        dVec force(sim.natoms);
        std::vector<int> permutation;
        std::mt19937 generator(993u);
        const int repeats = std::max(4, 20000 / sim.natoms);
        for (int alg = 0; alg < 2; ++alg) {
            BosonicExchangeBase& exchange = alg == 0 ? static_cast<BosonicExchangeBase&>(quad) : ladder;
            for (int round = 0; round < 5; ++round) {
                exchange.prepare(); exchange.exteriorSpringForce(force);
                MPI_Barrier(MPI_COMM_WORLD);
                double checksum = 0;
                double start = MPI_Wtime();
                for (int rep = 0; rep < repeats; ++rep) {
                    exchange.prepare();
                    exchange.exteriorSpringForce(force);
                    checksum += exchange.primEstimator() + force(0, 0);
                }
                double kernel = (MPI_Wtime()-start)*1e6/repeats;
                start = MPI_Wtime();
                for (int rep = 0; rep < repeats; ++rep) {
                    exchange.samplePermutation(permutation, generator);
                    checksum += permutation[0];
                }
                double sampler = (MPI_Wtime()-start)*1e6/repeats;
                double slow_kernel, slow_sampler;
                MPI_Reduce(&kernel, &slow_kernel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&sampler, &slow_sampler, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                struct rusage usage;
                getrusage(RUSAGE_SELF, &usage);
                if (rank == 0)
                    std::printf("TIMING,%d,%d,%s,%s,%d,%d,%.9g,%.9g,%ld,%.9g\n", NDIM,
                        sim.natoms, sim.pbc ? "free" : "trap", alg == 0 ? "quadratic" : "ladder", round, repeats,
                        slow_kernel, slow_sampler, usage.ru_maxrss, checksum);
            }
        }
    }
    MPI_Finalize();
    return 0;
}
