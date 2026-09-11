#pragma once

#include <random>
#include <vector>

class Simulation; // Forward declaration

/**
 * @class RelabelMove
 * @brief Monte Carlo relabelling of particles for bosonic simulations.
 *
 * The quadratic bosonic algorithm builds exchange cycles only over consecutive particle indices,
 * so the effective potential V_B depends on how the ring polymers are labelled. The partition
 * function is invariant under relabelling (all labellings integrate to the same Z and the physical
 * potential is permutation symmetric), but a molecular dynamics trajectory with fixed labels in a
 * dense liquid can only exchange neighbouring particles whose labels happen to be adjacent.
 *
 * In "metropolis" mode, the labels of two randomly chosen particles are swapped (all beads at once,
 * i.e. the rows of the coordinate, momentum and force arrays on every rank) and the swap is accepted
 * with probability min(1, exp(-beta_P * (V_B(new) - V_B(old)))). Interior springs, the physical
 * potential and the kinetic energy are invariant, so only V_B enters. This satisfies detailed balance
 * with respect to the sampled distribution and only improves ergodicity.
 *
 * In "shuffle" mode, a uniformly random permutation of the labels is applied without an acceptance
 * test, as done in Myung, Hirshberg, Parrinello, PRL 128, 045301 (2022) (SI Sec. III.B). This does
 * not preserve the sampled distribution exactly and is provided for comparison only.
 */
class RelabelMove {
public:
    RelabelMove(Simulation& _sim, const std::string& mode, int attempts, unsigned int seed);

    /// Attempts the move(s); must be called by all ranks at the same point of the MD step.
    void attempt();

    /// Cumulative acceptance ratio (rank 0).
    double acceptanceRatio() const;
    long numTrials() const { return n_trials; }
    long numAccepted() const { return n_accepted; }

    /// Current labels: labels[i] is the original index of the particle now stored in slot i.
    const std::vector<int>& labels() const { return current_labels; }

private:
    void applySwap(int i, int j);
    void attemptMetropolis();
    void attemptShuffle();

    Simulation& sim;
    bool metropolis;
    int n_attempts;
    std::mt19937 gen;   // Dedicated generator (rank 0 decides; decisions are broadcast)
    std::vector<int> current_labels;
    std::vector<int> buffer;
    long n_trials;
    long n_accepted;
};
