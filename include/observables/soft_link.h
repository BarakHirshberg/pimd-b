#pragma once

#include <string>

#include "observables/observable.h"

class Simulation;

/**
 * @class SoftLinkObservable
 * @brief Reports the state of the softened-link expanded ensemble (see SoftLinkMove).
 *
 * Columns: `gamma` (the current softening factor; estimators are valid only in the gamma = 1 stratum),
 * `soft_particle` (which particle carries the softened exterior link) and `soft_acc` (the cumulative
 * acceptance of the ladder jumps). Analysis keeps the rows with gamma = 1 and discards the rest.
 */
class SoftLinkObservable : public Observable {
public:
    SoftLinkObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);
    void calculate() override;
};
