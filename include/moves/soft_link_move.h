#pragma once

#include <random>
#include <string>
#include <vector>

class Simulation;

/**
 * @class SoftLinkMove
 * @brief Expanded ensemble over the stiffness of ONE exterior ring-polymer link ("continuous worm").
 *
 * Motivation. For hard-core particles the free-energy barrier between permutation sectors is the spring
 * cost of bringing two beads as close as the pair potential allows, about 0.26 P T k_BT for 2D helium
 * (see docs/09 of the benchmark repository). Softening the exterior spring leaving one particle by a
 * factor gamma divides that barrier by gamma. One softened link is not enough on its own: a transposition
 * of two particles replaces TWO exterior links, so half of the barrier survives and no exchange appears
 * (measured for N = 6 helium at 1 K and P = 32: p_exch = 0 up to gamma = 32). The softened region is
 * therefore a BLOCK of nsoft consecutive particles, whose every internal closure is softened at once;
 * nsoft = natoms softens the whole exterior spring network and is the default.
 *
 * Ensemble. The sampled distribution is
 *     pi(gamma, l*, R) ~ w(gamma) exp(-beta_P [U(R) + E_interior(R) + V_B(R; gamma, l*)]),
 * with gamma taken from a fixed ladder (gamma = 1 first) and l* from the particles. Conditioning on
 * gamma = 1 gives exactly the bosonic PIMD distribution, whatever the weights w: estimators are
 * therefore accumulated at gamma = 1 only and need NO reweighting. The weights affect efficiency alone
 * and are learned on the fly (Wang-Landau style) so that the ladder is traversed.
 *
 * Dynamics. The move changes no coordinate and no momentum: it changes one number in the Hamiltonian.
 * Between jumps the propagation is ordinary PIMD, and at gamma = 1 it is exactly bosonic PIMD, so
 * trajectories harvested at gamma = 1 are physical (see ADR-009).
 *
 * MPI. Rank 0 proposes and decides (it holds the first beads and the bosonic recursion); the new state
 * is broadcast and applied on every rank, after which the forces are refreshed.
 */
class SoftLinkMove {
public:
    SoftLinkMove(Simulation& _sim, const std::vector<double>& ladder, double wl_step, unsigned int seed,
                 int nsoft, int start_rung = 0);

    /// Attempts the (gamma, l*) jumps; must be called by all ranks at the same point of the MD step.
    void attempt();

    [[nodiscard]] double acceptanceRatio() const;
    [[nodiscard]] double gamma() const;
    [[nodiscard]] int rung() const { return index; }
    [[nodiscard]] int particle() const { return soft_particle; }
    [[nodiscard]] const std::vector<double>& weights() const { return log_weights; }

private:
    /// Effective potential with a trial (gamma, l*), restoring the previous state afterwards.
    double trialPotential(int rung_index, int particle_index);

    Simulation& sim;
    std::vector<double> gammas;      // ladder, gammas[0] = 1 (the physical stratum)
    std::vector<double> log_weights; // ln w(gamma), learned on the fly
    std::vector<long> histogram;     // visits per rung since the last refinement
    double wl_step;                  // Wang-Landau increment, halved as the histogram flattens
    int index;                       // current rung
    int nsoft;                       // number of consecutive particles whose links are softened
    int soft_particle;               // first particle of the softened block
    std::mt19937 gen;
    long n_trials;
    long n_accepted;
};
