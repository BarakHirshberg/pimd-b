#pragma once

#include "observables/observable.h"

class Simulation; // Forward declaration

/**
 * @class ExchangeMoveObservable
 * @brief Cumulative acceptance ratio (exchange_acc) and number of trials (exchange_ntrial) of the
 * exchange (segment regrowth) move; values on the first time slice.
 */
class ExchangeMoveObservable : public Observable {
public:
    ExchangeMoveObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void calculate() override;
};
