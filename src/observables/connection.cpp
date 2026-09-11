#include "observables/connection.h"
#include "simulation.h"

#include <algorithm>

/**
 * @brief Connection observable class constructor.
 */
ConnectionObservable::ConnectionObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "conn_mean", "conn_max", "p_exch_exact" });
}

/**
 * @brief Evaluates the connection diagnostics on the first time slice (other ranks contribute zero).
 */
void ConnectionObservable::calculate() {
    if (!(sim.this_bead == 0 && sim.bosonic)) {
        return;
    }

    const int n = sim.natoms;
    double sum_next = 0.0;
    double max_next = 0.0;
    double sum_self = 0.0;

    for (int l = 0; l < n; ++l) {
        sum_self += sim.bosonic_exchange->getConnectionProbability(l, l);
        if (l + 1 < n) {
            const double p = sim.bosonic_exchange->getConnectionProbability(l, l + 1);
            sum_next += p;
            max_next = std::max(max_next, p);
        }
    }

    quantities["conn_mean"] = (n > 1) ? sum_next / (n - 1) : 0.0;
    quantities["conn_max"] = max_next;
    quantities["p_exch_exact"] = 1.0 - sum_self / n;
}
