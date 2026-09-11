#include "moves/timeshift_move.h"
#include "simulation.h"
#include "mpi.h"

#include <cmath>
#include <vector>

TimeShiftMove::TimeShiftMove(Simulation& _sim, unsigned int seed) :
    sim(_sim), gen(seed), n_trials(0), n_accepted(0) {
    if (_sim.nbeads < 2) {
        throw std::invalid_argument("The time-shift move requires at least 2 beads.");
    }
}

/**
 * @brief Sends this rank's bead (coordinates and momenta) to rank (rank + shift) mod P and receives
 * the bead of rank (rank - shift) mod P, then refreshes the neighbour coordinates.
 */
void TimeShiftMove::rotate(int shift) {
    const int P = sim.nbeads;
    const int dest = (sim.this_bead + shift) % P;
    const int src = (sim.this_bead - shift + P) % P;
    const int n = sim.coord.size();

    std::vector<double> send(2 * n), recv(2 * n);
    std::copy(sim.coord.data(), sim.coord.data() + n, send.begin());
    std::copy(sim.momenta.data(), sim.momenta.data() + n, send.begin() + n);

    MPI_Sendrecv(send.data(), 2 * n, MPI_DOUBLE, dest, 1, recv.data(), 2 * n, MPI_DOUBLE, src, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::copy(recv.begin(), recv.begin() + n, sim.coord.data());
    std::copy(recv.begin() + n, recv.end(), sim.momenta.data());

    sim.updateNeighboringCoordinates();
}

/**
 * @brief Total classical spring energy: interior springs of ranks 1..P-1 plus the bosonic exterior
 * potential V_B (rank 0). The result is valid on rank 0 only.
 */
double TimeShiftMove::totalSpringEnergy() const {
    double local = 0.0;
    if (sim.this_bead != 0) {
        local = sim.classicalSpringEnergy();
    } else {
        sim.bosonic_exchange->prepare();
        local = sim.bosonic_exchange->effectivePotential();
    }
    double total = 0.0;
    MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total;
}

void TimeShiftMove::attempt() {
    const int P = sim.nbeads;
    int shift = 1;
    if (sim.this_bead == 0) {
        std::uniform_int_distribution<int> pick(1, P - 1);
        shift = pick(gen);
    }
    MPI_Bcast(&shift, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const double e_old = totalSpringEnergy();
    rotate(shift);
    const double e_new = totalSpringEnergy();

    int accept = 0;
    if (sim.this_bead == 0) {
        const double delta = e_new - e_old;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        accept = (delta <= 0.0 || uniform(gen) < std::exp(-sim.thermo_beta * delta)) ? 1 : 0;
        n_trials += 1;
        n_accepted += accept;
    }
    MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!accept) {
        rotate(P - shift);
    }
    // Forces (and the exchange state on the bosonic ranks) correspond to the current bead assignment
    sim.updateForces();
    if (accept) {
        sim.configurationChanged();
    }
}

double TimeShiftMove::acceptanceRatio() const {
    return (n_trials > 0) ? static_cast<double>(n_accepted) / n_trials : 0.0;
}
