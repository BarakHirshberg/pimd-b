#include "observables/soft_link.h"
#include "simulation.h"
#include "moves/soft_link_move.h"

SoftLinkObservable::SoftLinkObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "gamma", "soft_particle", "soft_acc" });
}

void SoftLinkObservable::calculate() {
    if (sim.this_bead != 0 || !sim.soft_link_move) {
        return;
    }
    quantities["gamma"] = sim.soft_link_move->gamma();
    quantities["soft_particle"] = static_cast<double>(sim.soft_link_move->particle());
    quantities["soft_acc"] = sim.soft_link_move->acceptanceRatio();
}
