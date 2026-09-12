#include "observables/soft_link.h"
#include "simulation.h"
#include "moves/soft_link_move.h"

SoftLinkObservable::SoftLinkObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "gamma", "soft_particle", "soft_acc" });
}

void SoftLinkObservable::calculate() {
    if (sim.this_bead != 0) {
        return;
    }
    if (!sim.soft_link_move) {
        // The observable is on but the move is off: the whole run is the physical ensemble, so report
        // gamma = 1 rather than 0, which keeps the "condition on gamma = 1" analysis meaningful.
        quantities["gamma"] = 1.0;
        return;
    }
    quantities["gamma"] = sim.soft_link_move->gamma();
    quantities["soft_particle"] = static_cast<double>(sim.soft_link_move->particle());
    quantities["soft_acc"] = sim.soft_link_move->acceptanceRatio();
}
