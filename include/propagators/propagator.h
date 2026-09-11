#pragma once

#include "common.h"

class Simulation;

class Propagator {
public:
    explicit Propagator(Simulation& _sim);
    virtual ~Propagator() = default;
    
    virtual void step() = 0;

    /**
     * @brief Re-synchronise any force arrays cached by the propagator after the configuration was
     * changed outside of step() (Monte Carlo label or exchange moves). Default: nothing cached.
     */
    virtual void refreshForces() {}
    void momentStep();
    void coordsStep();

protected:
    Simulation& sim; // Reference to the simulation object
};