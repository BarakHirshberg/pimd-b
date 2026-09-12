#include "moves/soft_link_move.h"
#include "simulation.h"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <numeric>

SoftLinkMove::SoftLinkMove(Simulation& _sim, const std::vector<double>& ladder, double _wl_step,
                           unsigned int seed, const int start_rung) :
    sim(_sim),
    gammas(ladder),
    log_weights(ladder.size(), 0.0),
    histogram(ladder.size(), 0),
    wl_step(_wl_step),
    index(start_rung),
    soft_particle(0),
    gen(seed),
    n_trials(0),
    n_accepted(0) {
    if (gammas.empty() || gammas.front() != 1.0) {
        throw std::invalid_argument("The softened-link ladder must start at gamma = 1 (the physical state)!");
    }
    if (index < 0 || index >= static_cast<int>(gammas.size())) {
        throw std::invalid_argument("soft_link_start_rung is outside the softened-link ladder!");
    }
    sim.soft_link_particle = soft_particle;
    sim.soft_link_gamma = gammas[index];
}

double SoftLinkMove::gamma() const {
    return gammas[index];
}

/**
 * @brief Bosonic effective potential for a trial (rung, particle); the caller restores the state.
 */
double SoftLinkMove::trialPotential(const int rung_index, const int particle_index) {
    sim.soft_link_particle = particle_index;
    sim.soft_link_gamma = gammas[rung_index];
    sim.bosonic_exchange->prepare();
    return sim.bosonic_exchange->effectivePotential();
}

void SoftLinkMove::attempt() {
    int decision[2] = {index, soft_particle};

    if (sim.this_bead == 0) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::uniform_int_distribution<int> pick_particle(0, sim.natoms - 1);

        const double v_old = trialPotential(index, soft_particle);

        // 1) jump one rung of the ladder (reflecting at the ends, hence symmetric)
        const int nrungs = static_cast<int>(gammas.size());
        if (nrungs > 1) {
            const int proposed = (uniform(gen) < 0.5) ? index - 1 : index + 1;
            if (proposed >= 0 && proposed < nrungs) {
                const double v_new = trialPotential(proposed, soft_particle);
                const double log_acc = -sim.thermo_beta * (v_new - v_old)
                                       + (log_weights[proposed] - log_weights[index]);
                n_trials += 1;
                if (log_acc >= 0.0 || uniform(gen) < std::exp(log_acc)) {
                    index = proposed;
                    n_accepted += 1;
                } else {
                    trialPotential(index, soft_particle);  // restore
                }
            }
        }

        // 2) move the softened link to another particle (uniform, hence symmetric). At gamma = 1 this
        //    changes nothing and is always accepted.
        const double v_cur = sim.bosonic_exchange->effectivePotential();
        const int proposed_particle = pick_particle(gen);
        if (proposed_particle != soft_particle) {
            const double v_new = trialPotential(index, proposed_particle);
            const double log_acc = -sim.thermo_beta * (v_new - v_cur);
            if (log_acc >= 0.0 || uniform(gen) < std::exp(log_acc)) {
                soft_particle = proposed_particle;
            } else {
                trialPotential(index, soft_particle);  // restore
            }
        }

        // Wang-Landau update of the ladder weights (efficiency only; the gamma = 1 stratum is exact
        // whatever the weights). The increment is halved once every rung has been visited often enough.
        log_weights[index] -= wl_step;
        histogram[index] += 1;
        const long total = std::accumulate(histogram.begin(), histogram.end(), 0L);
        const long lowest = *std::min_element(histogram.begin(), histogram.end());
        if (total > 200L * static_cast<long>(gammas.size())
            && lowest > (8L * total) / (10L * static_cast<long>(gammas.size()))) {
            wl_step *= 0.5;
            std::fill(histogram.begin(), histogram.end(), 0L);
        }
        const double shift = log_weights[0];
        for (double& w : log_weights) {
            w -= shift;  // keep ln w(gamma = 1) = 0 so the numbers stay readable
        }

        decision[0] = index;
        decision[1] = soft_particle;
    }

    MPI_Bcast(decision, 2, MPI_INT, 0, MPI_COMM_WORLD);
    index = decision[0];
    soft_particle = decision[1];
    sim.soft_link_particle = soft_particle;
    sim.soft_link_gamma = gammas[index];

    sim.updateForces();
    sim.configurationChanged();
}

double SoftLinkMove::acceptanceRatio() const {
    return (n_trials > 0) ? static_cast<double>(n_accepted) / n_trials : 0.0;
}
