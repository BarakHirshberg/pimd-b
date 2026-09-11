#pragma once

#include <vector>
#include <memory>
#include "thermostats/thermostat.h"

class Simulation;
class Coupling;

class LangevinThermostat : public Thermostat {
public:
    LangevinThermostat(Simulation& _sim, bool normal_modes);
    ~LangevinThermostat() override = default;

    void momentaUpdate() override;

protected:
    double friction_coefficient, noise_coefficient;
};

/**
 * @class PILEThermostat
 * @brief Path-integral Langevin equation (PILE-L) thermostat: the normal mode k held by this rank is
 * damped with friction gamma_k = 2 omega_k, omega_k = 2 omega_P sin(pi k / P), and the centroid (k = 0)
 * with the user friction gamma. Requires nmthermostat = true. For bosons the modes are those of the
 * separate (identity) ring polymers.
 */
class PILEThermostat : public LangevinThermostat {
public:
    PILEThermostat(Simulation& _sim, bool normal_modes);
    ~PILEThermostat() override = default;
};
