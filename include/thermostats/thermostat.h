#pragma once

#include <vector>
#include <memory>

class Simulation;
class Coupling;

class Thermostat {
public:
    explicit Thermostat(Simulation& _sim, bool normal_modes);
    virtual ~Thermostat() = default;
    void step();
    virtual void momentaUpdate();
    virtual double getAdditionToH();

    /**
     * Exchanges any per-particle thermostat state between particles i and j (used when particles
     * are relabelled). The default is a no-op, appropriate for thermostats without per-particle state.
     */
    virtual void swapParticles(int i, int j) {}
protected:
    Simulation& sim;   // Reference to the simulation object
    std::unique_ptr<Coupling> coupling;
};