#pragma once

#include "states/state.h"

class Simulation; // Forward declaration

/* -------------------------------- */

/**
 * @class LabelState
 * @brief Writes the current particle labels (original index of the particle stored in each slot)
 * to output/labels.dat on the first time slice, so trajectories can be unscrambled after
 * relabelling moves. Without the relabelling move the labels are the identity.
 */
class LabelState : public State {
public:
    LabelState(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void initialize() override;
    void output(int step) override;
};
