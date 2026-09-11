#pragma once

#include <random>

class Simulation; // Forward declaration

/**
 * @class TimeShiftMove
 * @brief Monte Carlo move that cyclically renumbers the imaginary-time slices (beads) of all particles,
 * moving the slice boundary at which bosonic exchange is evaluated, without changing any bead position
 * or momentum.
 *
 * The quantum trace is cyclic, so the exchange closures may be placed at any slice boundary; for a
 * given configuration, however, the effective potential (interior springs plus the bosonic exterior
 * potential V_B) depends on where the boundary sits. A shift by s beads is therefore proposed and
 * accepted with probability min(1, exp(-beta_P dH)), dH being the change of the total classical
 * spring energy (interior springs + V_B); the kinetic energy and the physical potential are invariant.
 * Combined with MD this satisfies detailed balance and lets the exchange closure be attempted at every
 * slice pair instead of a fixed one. The dynamics between moves is untouched (labels only).
 *
 * Implementation (one MPI rank per bead): all ranks send their coordinates and momenta to rank
 * (rank + s) mod P; on rejection the shift is undone. Thermostat state stays with the rank: the joint
 * target distribution is symmetric under reassigning identical thermostat chains to beads.
 */
class TimeShiftMove {
public:
    TimeShiftMove(Simulation& _sim, unsigned int seed);

    /// Attempts one shift; must be called by all ranks at the same point of the MD step.
    void attempt();

    double acceptanceRatio() const;
    long numTrials() const { return n_trials; }
    long numAccepted() const { return n_accepted; }

private:
    void rotate(int shift);
    double totalSpringEnergy() const;

    Simulation& sim;
    std::mt19937 gen;
    long n_trials;
    long n_accepted;
};
