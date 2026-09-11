#pragma once

#include "observables/observable.h"

class Simulation; // Forward declaration

/**
 * @class TimeShiftObservable
 * @brief Cumulative acceptance ratio (timeshift_acc) and number of trials (timeshift_ntrial) of the
 * imaginary-time shift move; values on the first time slice.
 */
class TimeShiftObservable : public Observable {
public:
    TimeShiftObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void calculate() override;
};
