// Finite-difference and sampler tests of the exchange classes on a random configuration.
//   mpirun -np P ladder_force_test <config.ini> [sigma_angstrom]
// Run from a scratch directory (the Simulation writes its output folder there). Ranks 0 and P-1
// each test their own force (first beads / last beads) against central differences of their own
// effective potential, for the quadratic class (control) and the ladder class, with and without a
// softened block. Then: sampled link frequencies against the connection probabilities, the sum
// rules, and (without winding sums) the primitive estimator against the sampled exterior energy.
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "mpi.h"
#include "params.h"
#include "simulation.h"
#include "bosonic_exchange/quadratic_bosonic_exchange.h"
#include "bosonic_exchange/ladder_bosonic_exchange.h"

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

// Interior part of the exterior-force array: k (x_neighbour - x) with the winding mean separation
void interiorTerm(const Simulation& sim, int l, double out[NDIM]) {
    const dVec& nb = (sim.this_bead == 0) ? sim.next_coord : sim.prev_coord;
    double d[NDIM];
    for (int axis = 0; axis < NDIM; ++axis) {
        double dx = nb(l, axis) - sim.coord(l, axis);
        if (sim.pbc) applyMinimumImage(dx, sim.size);
        d[axis] = dx;
    }
    if (sim.winding_springs) sim.linkMeanSeparation(d, d);
    for (int axis = 0; axis < NDIM; ++axis) out[axis] = sim.spring_constant * d[axis];
}

double maxRelForceError(Simulation& sim, BosonicExchangeBase& ex, const std::string& tag) {
    const int n = sim.natoms;
    ex.prepare();
    dVec f(n);
    ex.exteriorSpringForce(f);
    const double h = 1e-4;  // atomic units; links are O(1) bohr
    double worst = 0.0, scale = 0.0;
    for (int l = 0; l < n; ++l) {
        double interior[NDIM];
        interiorTerm(sim, l, interior);
        for (int axis = 0; axis < NDIM; ++axis) {
            const double x0 = sim.coord(l, axis);
            sim.coord(l, axis) = x0 + h; ex.prepare(); const double vp = ex.effectivePotential();
            sim.coord(l, axis) = x0 - h; ex.prepare(); const double vm = ex.effectivePotential();
            sim.coord(l, axis) = x0;
            const double fd = -(vp - vm) / (2 * h);
            const double fa = f(l, axis) - interior[axis];
            worst = std::max(worst, std::abs(fd - fa));
            scale = std::max(scale, std::abs(fa));
        }
    }
    ex.prepare();
    const double rel = worst / std::max(scale, 1e-300);
    std::printf("rank %d %-28s max |F_fd - F| = %.3e, max |F| = %.3e, rel = %.3e\n", sim.this_bead, tag.c_str(), worst, scale, rel);
    return rel;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) { if (rank == 0) std::printf("usage: ladder_force_test <config.ini> [sigma_angstrom]\n"); MPI_Finalize(); return 2; }
    const double sigma_bohr = ((argc > 2) ? std::stod(argv[2]) : 1.0) / 0.529177210903;

    {
        Params params(argv[1], rank);
        Simulation sim(rank, size, params, 4242u);
        const int n = sim.natoms;

        // Random configuration: rank 0 draws particle centres uniformly in the box; every rank adds
        // its own Gaussian jitter so that beads of one ring stay within a link length of each other.
        std::vector<double> centres(n * NDIM);
        if (rank == 0) {
            std::mt19937 g(99);
            std::uniform_real_distribution<double> u(-0.5 * sim.size, 0.5 * sim.size);
            for (auto& c : centres) c = u(g);
        }
        MPI_Bcast(centres.data(), n * NDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::mt19937 gj(1000 + rank);
        std::normal_distribution<double> jitter(0.0, sigma_bohr);
        for (int l = 0; l < n; ++l)
            for (int axis = 0; axis < NDIM; ++axis)
                sim.coord(l, axis) = centres[l * NDIM + axis] + jitter(gj);
        sim.updateNeighboringCoordinates();

        if (sim.is_bosonic_bead) {
            BosonicExchange quad(sim);
            LadderBosonicExchange lad(sim);

            // Forces vs finite differences, plain
            sim.soft_link_count = 0; sim.soft_link_gamma = 1.0; sim.soft_link_particle = 0;
            check(maxRelForceError(sim, quad, "quadratic, gamma = 1") < 1e-6, "quadratic FD plain");
            check(maxRelForceError(sim, lad, "ladder, gamma = 1") < 1e-6, "ladder FD plain");
            std::printf("rank %d V_quadratic = %.10e  V_ladder = %.10e (differ pointwise by construction)\n",
                        rank, quad.effectivePotential(), lad.effectivePotential());

            // With a softened block of 5 particles starting at 3, gamma = 4
            sim.soft_link_particle = 3; sim.soft_link_count = 5; sim.soft_link_gamma = 4.0;
            check(maxRelForceError(sim, quad, "quadratic, block 5 gamma 4") < 1e-6, "quadratic FD soft");
            check(maxRelForceError(sim, lad, "ladder, block 5 gamma 4") < 1e-6, "ladder FD soft");
            sim.soft_link_count = 0; sim.soft_link_gamma = 1.0; sim.soft_link_particle = 0;
            quad.prepare(); lad.prepare();

            if (rank == 0) {
                // Sum rules and sampler vs connection probabilities (both classes)
                std::mt19937 gen(31337);
                const int nsamp = 200000;
                for (int which = 0; which < 2; ++which) {
                    BosonicExchangeBase& ex = (which == 0) ? static_cast<BosonicExchangeBase&>(quad) : lad;
                    const std::string tag = (which == 0) ? "quadratic" : "ladder";
                    for (int l = 0; l < n; ++l) {
                        double out = 0.0, in = 0.0;
                        for (int u = 0; u < n; ++u) { out += ex.getConnectionProbability(l, u); in += ex.getConnectionProbability(u, l); }
                        check(std::abs(out - 1.0) < 1e-8, tag + " outgoing sum l = " + std::to_string(l));
                        check(std::abs(in - 1.0) < 1e-8, tag + " incoming sum l = " + std::to_string(l));
                    }
                    std::map<std::pair<int, int>, int> counts;
                    std::vector<int> perm;
                    double e_ext_mean = 0.0;
                    for (int s = 0; s < nsamp; ++s) {
                        ex.samplePermutation(perm, gen);
                        std::vector<int> hit(n, 0);
                        for (int l = 0; l < n; ++l) { counts[{l, perm[l]}] += 1; hit[perm[l]] += 1; }
                        for (int l = 0; l < n; ++l) check(hit[l] == 1, tag + " sample is a permutation");
                        // exterior spring energy of this permutation (minimum image, no winding sums)
                        double e = 0.0;
                        for (int l = 0; l < n; ++l) {
                            double d2 = 0.0;
                            for (int axis = 0; axis < NDIM; ++axis) {
                                double dx = sim.coord(perm[l], axis) - sim.prev_coord(l, axis);
                                if (sim.pbc) applyMinimumImage(dx, sim.size);
                                d2 += dx * dx;
                            }
                            e += 0.5 * sim.linkStiffness(l) * d2;
                        }
                        e_ext_mean += e;
                    }
                    e_ext_mean /= nsamp;
                    double worst = 0.0;
                    for (int l = 0; l < n; ++l) {
                        for (int u = 0; u < n; ++u) {
                            const double p = ex.getConnectionProbability(l, u);
                            const double f = static_cast<double>(counts[{l, u}]) / nsamp;
                            const double sg = std::sqrt(std::max(p * (1 - p), 1e-12) / nsamp);
                            worst = std::max(worst, std::abs(f - p) / sg);
                            check(std::abs(f - p) < 5 * sg + 2e-4, tag + " link frequency " + std::to_string(l) + "->" + std::to_string(u)
                                  + ": " + std::to_string(f) + " vs " + std::to_string(p));
                        }
                    }
                    std::printf("rank 0 %s sampler vs connection probabilities: worst %.2f sigma over %d links\n", tag.c_str(), worst, n * n);
                    if (!sim.winding_springs) {
                        double prim = ex.primEstimator();
#if IPI_CONVENTION
                        prim *= sim.nbeads;
#endif
                        const double se = 0.0;  // the sampled mean has its own noise; 1 % tolerance
                        (void)se;
                        std::printf("rank 0 %s primEstimator = %.8e, -<E_ext> sampled = %.8e\n", tag.c_str(), prim, -e_ext_mean);
                        check(std::abs(prim + e_ext_mean) < 0.01 * std::abs(e_ext_mean) + 1e-10, tag + " primitive estimator vs sampled exterior energy");
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int total = 0;
    MPI_Reduce(&failures, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total == 0) std::printf("ladder_force_test: all checks passed\n");
        else std::printf("ladder_force_test: %d failures\n", total);
    }
    MPI_Finalize();
    return (rank == 0 && total != 0) ? 1 : 0;
}
