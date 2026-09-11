#include "observables/permutation.h"
#include "simulation.h"
#include "units.h"
#include "mpi.h"

#include <algorithm>
#include <array>
#include <fstream>

/**
 * @brief Permutation observable class constructor.
 *
 * @param _winding If true, also evaluates the total winding vector and the superfluid fraction (requires PBC).
 */
PermutationObservable::PermutationObservable(const Simulation& _sim, int _freq, const std::string& _out_unit, bool _winding) :
    Observable(_sim, _freq, _out_unit),
    winding(_winding),
    nsamples(_sim.n_perm_samples),
    gen(_sim.params_seed + 7919),  // Dedicated stream: must not perturb the thermostat random numbers
    perm(_sim.natoms),
    hist(_sim.natoms, 0),
    total_samples(0),
    hist_filename(std::format("{}/cycles.dat", Output::FOLDER_NAME)) {
    if (winding && !sim.pbc) {
        throw std::invalid_argument("The winding observable requires periodic boundary conditions (pbc = true).");
    }

    if (winding) {
        initialize({ "n_cycles", "cycle_max", "p_exch", "w2", "rho_s" });
    } else {
        initialize({ "n_cycles", "cycle_max", "p_exch" });
    }
}

PermutationObservable::~PermutationObservable() = default;

/**
 * @brief Samples permutations on the first time slice and accumulates cycle statistics.
 * With winding enabled, all ranks participate in a reduction of the interior displacements.
 */
void PermutationObservable::calculate() {
    if (!sim.bosonic) {
        return;
    }

    // Interior springs: displacement from this bead to the next one, summed over particles
    // (the last time slice is excluded because its next bead is the exterior link)
    std::array<double, NDIM> interior_local = {};
    std::array<double, NDIM> interior_total = {};

    if (winding) {
        if (sim.this_bead != sim.nbeads - 1) {
            for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    double diff = sim.next_coord(ptcl_idx, axis) - sim.coord(ptcl_idx, axis);
                    applyMinimumImage(diff, sim.size);
                    interior_local[axis] += diff;
                }
            }
        }

        MPI_Reduce(interior_local.data(), interior_total.data(), NDIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (sim.this_bead != 0) {
        return;
    }

    double n_cycles = 0.0, cycle_max = 0.0, p_exch = 0.0, w2 = 0.0;
    std::vector<int> visited(sim.natoms);

    for (int s = 0; s < nsamples; ++s) {
        sim.bosonic_exchange->samplePermutation(perm, gen);

        // Cycle decomposition
        std::fill(visited.begin(), visited.end(), 0);
        int longest = 0, ncyc = 0, in_exchange = 0;
        for (int start = 0; start < sim.natoms; ++start) {
            if (visited[start]) continue;
            int length = 0;
            for (int l = start; !visited[l]; l = perm[l]) {
                visited[l] = 1;
                ++length;
            }
            ++ncyc;
            longest = std::max(longest, length);
            if (length >= 2) in_exchange += length;
            hist[length - 1] += 1;
        }
        ++total_samples;
        n_cycles += ncyc;
        cycle_max += longest;
        p_exch += static_cast<double>(in_exchange) / sim.natoms;

        if (winding) {
            // Exterior links: last bead (prev_coord on rank 0) of l -> first bead (coord) of perm[l]
            std::array<double, NDIM> w = interior_total;
            for (int l = 0; l < sim.natoms; ++l) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    double diff = sim.coord(perm[l], axis) - sim.prev_coord(l, axis);
                    applyMinimumImage(diff, sim.size);
                    w[axis] += diff;
                }
            }
            double w2_sample = 0.0;
            for (int axis = 0; axis < NDIM; ++axis) {
                w2_sample += w[axis] * w[axis];
            }
            w2 += w2_sample;
        }
    }

    quantities["n_cycles"] = n_cycles / nsamples;
    quantities["cycle_max"] = cycle_max / nsamples;
    quantities["p_exch"] = p_exch / nsamples;

    if (winding) {
        w2 /= nsamples;
        // rho_s/rho = m <W^2> / (d hbar^2 beta N) = <W^2> / (2 lambda beta N d), lambda = hbar^2/(2m)
        const double lambda = Constants::hbar * Constants::hbar / (2.0 * sim.mass);
        quantities["rho_s"] = w2 / (2.0 * lambda * sim.beta * sim.natoms * NDIM);
        const double len = Units::convertToUser("length", out_unit.empty() ? "atomic_unit" : out_unit, 1.0);
        quantities["w2"] = w2 * len * len;
    }

    if (sim.getStep() % sim.sfreq == 0) {
        writeHistogram();
    }
}

/**
 * @brief Overwrites output/cycles.dat with the cumulative cycle-length distribution.
 */
void PermutationObservable::writeHistogram() {
    std::ofstream file(hist_filename, std::ios::out | std::ios::trunc);
    file << std::format("{:>8s} {:>14s} {:>14s}\n", "length", "count", "P(length)");
    for (int k = 1; k <= sim.natoms; ++k) {
        const double prob = (total_samples > 0) ? static_cast<double>(hist[k - 1]) / total_samples : 0.0;
        file << std::format("{:>8d} {:>14d} {:>14.8e}\n", k, hist[k - 1], prob);
    }
}
