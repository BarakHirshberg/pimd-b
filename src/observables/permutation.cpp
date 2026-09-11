#include "observables/permutation.h"
#include "simulation.h"
#include "units.h"
#include "winding.h"
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
    link_gen(_sim.params_seed + 7919 + 104729u * static_cast<unsigned int>(_sim.this_bead + 1)),
    perm(_sim.natoms),
    hist(_sim.natoms, 0),
    gperm(_sim.natoms),
    total_samples(0),
    hist_filename(std::format("{}/cycles.dat", Output::FOLDER_NAME)) {
    if (winding && !sim.pbc) {
        throw std::invalid_argument("The winding observable requires periodic boundary conditions (pbc = true).");
    }

    if (winding) {
        initialize({ "n_cycles", "cycle_max", "p_exch", "w2", "rho_s", "p_exch_geom", "w2_geom", "rho_s_geom" });
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
            std::array<double, NDIM> link = {};
            for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
                linkVector(sim.coord, ptcl_idx, sim.next_coord, ptcl_idx, link);
                for (int axis = 0; axis < NDIM; ++axis) {
                    interior_local[axis] += link[axis];
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
            std::array<double, NDIM> link = {};
            for (int l = 0; l < sim.natoms; ++l) {
                linkVector(sim.prev_coord, l, sim.coord, perm[l], link);
                for (int axis = 0; axis < NDIM; ++axis) {
                    w[axis] += link[axis];
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
        const double rho_s_norm = 1.0 / (2.0 * lambda * sim.beta * sim.natoms * NDIM);
        quantities["rho_s"] = w2 * rho_s_norm;
        const double len = Units::convertToUser("length", out_unit.empty() ? "atomic_unit" : out_unit, 1.0);
        quantities["w2"] = w2 * len * len;

        // Approximate geometric reconstruction (PRL 2022 SI Alg. 1), deterministic given the configuration
        geometricPermutation();
        std::array<double, NDIM> wg = interior_total;
        std::array<double, NDIM> link = {};
        int in_exchange_geom = 0;
        for (int l = 0; l < sim.natoms; ++l) {
            if (gperm[l] != l) ++in_exchange_geom;
            linkVector(sim.prev_coord, l, sim.coord, gperm[l], link);
            for (int axis = 0; axis < NDIM; ++axis) {
                wg[axis] += link[axis];
            }
        }
        double w2_geom = 0.0;
        for (int axis = 0; axis < NDIM; ++axis) {
            w2_geom += wg[axis] * wg[axis];
        }
        quantities["p_exch_geom"] = static_cast<double>(in_exchange_geom) / sim.natoms;
        quantities["w2_geom"] = w2_geom * len * len;
        quantities["rho_s_geom"] = w2_geom * rho_s_norm;
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

/**
 * @brief Geometric permutation of Myung, Hirshberg & Parrinello (PRL 2022, SI Sec. III.C, Alg. 1).
 *
 * The last bead of every particle l is joined to the nearest (minimum-image) first bead among all
 * particles. Conflicts (two last beads choosing the same first bead) are resolved greedily in particle
 * order by reverting the later one to its own first bead if that is still free; any last beads left
 * unassigned are finally matched to the remaining free first beads by nearest distance ("close any
 * open rings"). The result is a permutation, but not a sample of the bosonic ensemble: it is an
 * approximate estimator kept here only to test the published procedure against the exact one.
 */
void PermutationObservable::geometricPermutation() {
    const int n = sim.natoms;
    std::vector<int> used(n, 0);
    std::fill(gperm.begin(), gperm.end(), -1);

    auto dist2 = [&](int l, int j) {
        double d2 = 0.0;
        for (int axis = 0; axis < NDIM; ++axis) {
            double diff = sim.coord(j, axis) - sim.prev_coord(l, axis);
            applyMinimumImage(diff, sim.size);
            d2 += diff * diff;
        }
        return d2;
    };

    for (int l = 0; l < n; ++l) {
        int best = -1;
        double best_d2 = 0.0;
        for (int j = 0; j < n; ++j) {
            const double d2 = dist2(l, j);
            if (best < 0 || d2 < best_d2) {
                best = j;
                best_d2 = d2;
            }
        }
        if (used[best]) {
            best = used[l] ? -1 : l;  // Revert the exchange if possible
        }
        if (best >= 0) {
            gperm[l] = best;
            used[best] = 1;
        }
    }

    // Close open rings: match the remaining last beads to the remaining first beads (nearest first)
    for (int l = 0; l < n; ++l) {
        if (gperm[l] >= 0) continue;
        int best = -1;
        double best_d2 = 0.0;
        for (int j = 0; j < n; ++j) {
            if (used[j]) continue;
            const double d2 = dist2(l, j);
            if (best < 0 || d2 < best_d2) {
                best = j;
                best_d2 = d2;
            }
        }
        gperm[l] = best;
        used[best] = 1;
    }
}

/**
 * @brief Link vector from bead (from, l_from) to bead (to, l_to): the minimum-image separation, plus a
 * sampled image shift w L when the springs are winding-summed. With winding_springs the MD configuration
 * only fixes the minimum-image representative of every link; the physical winding of the link is
 * distributed according to p_w of WindingProbability, and sampling it makes W an exact estimator.
 */
void PermutationObservable::linkVector(const dVec& from, int l_from, const dVec& to, int l_to, std::array<double, NDIM>& out) {
    for (int axis = 0; axis < NDIM; ++axis) {
        double diff = to(l_to, axis) - from(l_from, axis);
        applyMinimumImage(diff, sim.size);
        if (sim.winding_springs) {
            const WindingProbability wp(diff, sim.max_wind, sim.beta_half_k, sim.size);
            diff += sim.size * wp.sample(link_gen);
        }
        out[axis] = diff;
    }
}
