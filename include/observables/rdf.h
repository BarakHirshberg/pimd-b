#pragma once

#include <vector>

#include "observables/observable.h"

class Simulation; // Forward declaration

/* -------------------------------- */

/**
 * @class RDFObservable
 * @brief Radial distribution function g(r) accumulated over all time slices and particle pairs
 * using minimum-image distances. Adds no column to simulation.out; the normalised histogram is
 * written to output/rdf.dat (columns: r, g(r), counts) every sfreq steps, so a running estimate is
 * available during long simulations.
 */
class RDFObservable : public Observable {
public:
    RDFObservable(const Simulation& _sim, int _freq, const std::string& _out_unit);

    void calculate() override;

private:
    void write();

    int nbins;
    double rmax;
    double dr;
    std::vector<double> hist;
    long nsamples;              // Number of configurations (time slices x steps) accumulated on this rank
    std::string filename;
};
