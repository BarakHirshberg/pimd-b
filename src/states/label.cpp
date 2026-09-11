#include "states/label.h"
#include "moves.h"
#include "simulation.h"

#include <numeric>

LabelState::LabelState(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    State(_sim, _freq, _out_unit) {
}

void LabelState::initialize() {
    if (sim.this_bead == 0) {
        out_file.open(std::format("{}/labels.dat", Output::FOLDER_NAME), std::ios::out | std::ios::app);
    }
}

void LabelState::output(int step) {
    if (sim.this_bead != 0 || step % freq != 0) {
        return;
    }
    out_file << step;
    if (sim.relabel) {
        for (int label : sim.relabel_move->labels()) {
            out_file << ' ' << label;
        }
    } else {
        for (int i = 0; i < sim.natoms; ++i) {
            out_file << ' ' << i;
        }
    }
    out_file << '\n';
}
