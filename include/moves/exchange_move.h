#pragma once

#include <array>
#include <random>
#include <vector>

#include "common.h"

class Simulation; // Forward declaration

/**
 * @class ExchangeMove
 * @brief Monte Carlo move that regrows the last m beads of two particles so that their ring polymers
 * close either onto themselves (identity target) or onto each other (exchange target).
 *
 * Motivation: in bosonic PIMD the exchange sector is encoded in the smooth effective potential V_B, but
 * a molecular dynamics trajectory of hard-core particles has to pass through configurations with a
 * stretched exterior link (or overlapping cores) to switch closure, a free-energy barrier that grows
 * with the number of beads. This move jumps directly between sectors, as the swap move of the worm
 * algorithm does in PIMC.
 *
 * Proposal: pick particles a != b (consecutive labels, which is where the quadratic algorithm assigns
 * exchange weight), a segment length m (1 <= m <= max_segment) and a target with probability 1/2:
 *  - identity: regrow beads P-m+1..P of a as a free-particle (Levy) bridge from a_{P-m} to a_1 and of
 *    b from b_{P-m} to b_1;
 *  - exchange: regrow a from a_{P-m} to b_1 and b from b_{P-m} to a_1.
 * The proposal density of a segment is the mixture over the two targets of the Levy bridge densities
 * (spring Boltzmann factors of the regrown links including the closing link, divided by the free
 * propagator of the anchor-to-target displacement). The interior spring energies cancel between the
 * target distribution and the proposal, so
 *
 *   A = min{1, exp(-beta_P [dU + dV_B]) * S(old) / S(new) },
 *   S(seg) = sum_tau exp(-beta_P [E_close,tau(seg) - g_tau]),  g_tau = k |end_tau - anchor|^2 / (2 (m+1)),
 *
 * where dU is the change of the physical potential on the regrown slices and dV_B the change of the
 * bosonic exterior potential. Detailed balance holds for the joint (positions, momenta) distribution;
 * momenta are untouched.
 *
 * Communication (one rank per bead): the two particles' full paths are gathered on all ranks
 * (MPI_Allgather of 2 x NDIM doubles per rank); rank 0 draws the proposal and broadcasts it; ranks
 * holding regrown beads evaluate dU for their slice (MPI_Reduce to rank 0); rank 0 evaluates dV_B with
 * the proposed bead-P positions and broadcasts the decision; accepted proposals are applied and the
 * forces recomputed.
 */
class ExchangeMove {
public:
    ExchangeMove(Simulation& _sim, int max_segment, int attempts, unsigned int seed);

    /// Attempts the move(s); must be called by all ranks at the same point of the MD step.
    void attempt();

    double acceptanceRatio() const;
    long numTrials() const { return n_trials; }
    long numAccepted() const { return n_accepted; }

private:
    void gatherPaths(int a, int b);
    void levyBridge(const std::array<double, NDIM>& start, const std::array<double, NDIM>& end, int m,
                    std::vector<std::array<double, NDIM>>& out);
    double slicePotentialChange(int a, int b, const std::array<double, NDIM>& new_a,
                                const std::array<double, NDIM>& new_b) const;
    void separation(const std::array<double, NDIM>& x1, const std::array<double, NDIM>& x2,
                    std::array<double, NDIM>& diff) const;

    Simulation& sim;
    int max_segment;
    int n_attempts;
    std::mt19937 gen;
    long n_trials;
    long n_accepted;

    // Full paths of the two chosen particles (index = bead), identical on all ranks after gatherPaths()
    std::vector<std::array<double, NDIM>> path_a;
    std::vector<std::array<double, NDIM>> path_b;
    std::vector<double> gather_buffer;
    std::vector<double> proposal_buffer;
};
