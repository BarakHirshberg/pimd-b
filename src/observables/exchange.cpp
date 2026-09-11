#include "observables/exchange.h"
#include "moves.h"
#include "simulation.h"

ExchangeMoveObservable::ExchangeMoveObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "exchange_acc", "exchange_ntrial" });
}

void ExchangeMoveObservable::calculate() {
    if (sim.this_bead != 0 || !sim.exchange_move) {
        return;
    }
    quantities["exchange_acc"] = sim.exchange_mc->acceptanceRatio();
    quantities["exchange_ntrial"] = static_cast<double>(sim.exchange_mc->numTrials());
}
