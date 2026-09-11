#pragma once

#include "observables/observable.h"

class Simulation; // Forward declaration

/* -------------------------------- */

/**
 * @class RelabelObservable
 * @brief Cumulative acceptance ratio (relabel_acc) and number of trials (relabel_ntrial) of the
 * particle relabelling move (values on the first time slice; other ranks contribute zero).
 */
class RelabelObservable : public Observable {
public:
    RelabelObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void calculate() override;
};
