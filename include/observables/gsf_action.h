#pragma once

#include "observables/observable.h"

class Simulation; // Forward declaration

/* -------------------------------- */

class GSFActionObservable : public Observable {
public:
    GSFActionObservable(const Simulation& _sim, int _freq, const std::string& _out_unit, bool _extra = false);

    void calculate() override;

private:
    bool extra;  // Also report even_pot_gsf and kin_gsf
};
