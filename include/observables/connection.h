#pragma once

#include "observables/observable.h"

class Simulation; // Forward declaration

/* -------------------------------- */

/**
 * @class ConnectionObservable
 * @brief Cheap diagnostics of exchange activity from the connection probabilities of the
 * quadratic bosonic algorithm (or their enumeration in the factorial algorithm).
 *
 * Quantities (dimensionless, evaluated on the first time slice):
 *  - conn_mean: mean over l of Pr(last bead of l -> first bead of l+1)
 *  - conn_max:  maximum of the same quantity
 *  - p_exch_exact: 1 - (1/N) sum_l Pr(l -> l), probability that a particle is not in a 1-cycle
 */
class ConnectionObservable : public Observable {
public:
    ConnectionObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void calculate() override;
};
