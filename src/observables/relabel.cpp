#include "observables/relabel.h"
#include "moves.h"
#include "simulation.h"

RelabelObservable::RelabelObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "relabel_acc", "relabel_ntrial" });
}

void RelabelObservable::calculate() {
    if (sim.this_bead != 0 || !sim.relabel) {
        return;
    }
    quantities["relabel_acc"] = sim.relabel_move->acceptanceRatio();
    quantities["relabel_ntrial"] = static_cast<double>(sim.relabel_move->numTrials());
}
