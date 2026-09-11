#include "moves/relabel_move.h"
#include "simulation.h"
#include "thermostats.h"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <numeric>

RelabelMove::RelabelMove(Simulation& _sim, const std::string& mode, int attempts, unsigned int seed) :
    sim(_sim),
    metropolis(mode == "metropolis"),
    n_attempts(attempts),
    gen(seed),
    current_labels(_sim.natoms),
    buffer(1 + 2 * std::max(attempts, _sim.natoms)),
    n_trials(0),
    n_accepted(0) {
    std::iota(current_labels.begin(), current_labels.end(), 0);
}

/**
 * @brief Swaps the rows i and j of all per-particle arrays on this rank (whole ring polymers are
 * relabelled because every rank performs the same swap).
 */
void RelabelMove::applySwap(int i, int j) {
    if (i == j) return;
    sim.coord.swapRows(i, j);
    sim.prev_coord.swapRows(i, j);
    sim.next_coord.swapRows(i, j);
    sim.momenta.swapRows(i, j);
    sim.forces.swapRows(i, j);
    sim.thermostat->swapParticles(i, j);
    std::swap(current_labels[i], current_labels[j]);
}

void RelabelMove::attempt() {
    if (metropolis) {
        attemptMetropolis();
    } else {
        attemptShuffle();
    }
}

/**
 * @brief Rank 0 performs the Metropolis trials on its own copy of beads 1 and P (coord and
 * prev_coord), undoing rejected swaps and finally all accepted ones, then broadcasts the accepted
 * swaps. All ranks apply them and recompute the forces (the bosonic exterior forces are not
 * invariant under relabelling).
 */
void RelabelMove::attemptMetropolis() {
    const int n = sim.natoms;
    std::vector<int> accepted;

    if (sim.this_bead == 0) {
        std::uniform_int_distribution<int> pick_i(0, n - 1);
        std::uniform_int_distribution<int> pick_j(0, n - 2);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double v_old = sim.bosonic_exchange->effectivePotential();

        for (int t = 0; t < n_attempts; ++t) {
            const int i = pick_i(gen);
            int j = pick_j(gen);
            if (j >= i) ++j;

            sim.coord.swapRows(i, j);
            sim.prev_coord.swapRows(i, j);
            sim.bosonic_exchange->prepare();
            const double v_new = sim.bosonic_exchange->effectivePotential();

            const double delta = v_new - v_old;
            const bool accept = (delta <= 0.0) || (uniform(gen) < std::exp(-sim.thermo_beta * delta));

            if (accept) {
                accepted.push_back(i);
                accepted.push_back(j);
                v_old = v_new;
            } else {
                sim.coord.swapRows(i, j);
                sim.prev_coord.swapRows(i, j);
            }
        }

        // Restore rank 0 to the common state before the collective application below
        for (int k = static_cast<int>(accepted.size()) - 2; k >= 0; k -= 2) {
            sim.coord.swapRows(accepted[k], accepted[k + 1]);
            sim.prev_coord.swapRows(accepted[k], accepted[k + 1]);
        }

        n_trials += n_attempts;
        n_accepted += static_cast<long>(accepted.size() / 2);
    }

    // Broadcast the list of accepted swaps: buffer = {count, i1, j1, i2, j2, ...}
    buffer[0] = static_cast<int>(accepted.size() / 2);
    std::copy(accepted.begin(), accepted.end(), buffer.begin() + 1);
    MPI_Bcast(buffer.data(), 1 + 2 * n_attempts, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_acc = buffer[0];
    for (int k = 0; k < n_acc; ++k) {
        applySwap(buffer[1 + 2 * k], buffer[2 + 2 * k]);
    }

    if (n_acc > 0) {
        sim.updateForces();
        sim.configurationChanged();
    } else if (sim.is_bosonic_bead) {
        // Rank 0 evaluated trial configurations; restore the exchange state of the actual one
        sim.bosonic_exchange->prepare();
    }
}

/**
 * @brief Applies a uniformly random permutation of the labels (no acceptance test; approximate).
 */
void RelabelMove::attemptShuffle() {
    const int n = sim.natoms;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    if (sim.this_bead == 0) {
        std::shuffle(perm.begin(), perm.end(), gen);
        n_trials += 1;
        n_accepted += 1;
    }
    MPI_Bcast(perm.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // Decompose the permutation into transpositions and apply them (slot k receives particle perm[k])
    std::vector<int> pos(n);       // pos[p] = current slot of original slot p
    std::vector<int> slot(n);      // slot[k] = original slot currently at k
    std::iota(pos.begin(), pos.end(), 0);
    std::iota(slot.begin(), slot.end(), 0);
    for (int k = 0; k < n; ++k) {
        const int wanted = perm[k];
        const int where = pos[wanted];
        if (where != k) {
            applySwap(k, where);
            const int displaced = slot[k];
            slot[k] = wanted; slot[where] = displaced;
            pos[wanted] = k; pos[displaced] = where;
        }
    }

    sim.updateForces();
    sim.configurationChanged();
}

double RelabelMove::acceptanceRatio() const {
    return (n_trials > 0) ? static_cast<double>(n_accepted) / n_trials : 0.0;
}
