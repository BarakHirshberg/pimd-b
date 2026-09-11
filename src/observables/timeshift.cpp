#include "observables/timeshift.h"
#include "moves.h"
#include "simulation.h"

TimeShiftObservable::TimeShiftObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "timeshift_acc", "timeshift_ntrial" });
}

void TimeShiftObservable::calculate() {
    if (sim.this_bead != 0 || !sim.timeshift) {
        return;
    }
    quantities["timeshift_acc"] = sim.timeshift_move->acceptanceRatio();
    quantities["timeshift_ntrial"] = static_cast<double>(sim.timeshift_move->numTrials());
}
