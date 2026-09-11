#include "observables/rdf.h"
#include "simulation.h"
#include "units.h"
#include "mpi.h"

#include <cmath>
#include <fstream>
#include <numbers>

/**
 * @brief RDF observable class constructor. The output unit is the length unit of the r column.
 */
RDFObservable::RDFObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit),
    nbins(_sim.rdf_bins),
    rmax(_sim.rdf_rmax),
    hist(_sim.rdf_bins, 0.0),
    nsamples(0),
    filename(std::format("{}/rdf.dat", Output::FOLDER_NAME)) {
    if (rmax <= 0.0) {
        rmax = 0.5 * sim.size;
    }
    if (sim.pbc) {
        rmax = std::min(rmax, 0.5 * sim.size);
    }
    dr = rmax / nbins;

    try {
        Units::convertToUser("length", out_unit, 1.0);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("Invalid output unit for the rdf observable (expected a length unit).");
    }
}

/**
 * @brief Accumulates minimum-image pair distances of the current time slice; every sfreq steps the
 * histograms of all ranks are reduced to rank 0 and written to disk.
 */
void RDFObservable::calculate() {
    for (int i = 0; i < sim.natoms; ++i) {
        for (int j = i + 1; j < sim.natoms; ++j) {
            const double r = sim.getSeparation(i, j, true).norm();
            if (r < rmax) {
                hist[static_cast<int>(r / dr)] += 1.0;
            }
        }
    }
    ++nsamples;

    if (sim.getStep() % sim.sfreq == 0) {
        write();
    }
}

/**
 * @brief Reduces the histograms over ranks and writes the normalised g(r) on rank 0.
 *
 * g(r_k) = H_k / (n_conf * N(N-1)/2 * V_shell(k)/V), with V = L^d and V_shell the d-dimensional
 * shell volume between r_k and r_k + dr. Without PBC, V = L^d is used nominally.
 */
void RDFObservable::write() {
    std::vector<double> total(nbins, 0.0);
    long total_samples = 0;
    MPI_Reduce(hist.data(), total.data(), nbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nsamples, &total_samples, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (sim.this_bead != 0) {
        return;
    }

    const double volume = std::pow(sim.size, NDIM);
    const double npairs = 0.5 * sim.natoms * (sim.natoms - 1);

    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    file << std::format("# r [{}]  g(r)  counts\n", out_unit);
    for (int k = 0; k < nbins; ++k) {
        const double r1 = k * dr, r2 = (k + 1) * dr;
        double shell;
        if constexpr (NDIM == 3) {
            shell = 4.0 / 3.0 * std::numbers::pi * (r2 * r2 * r2 - r1 * r1 * r1);
        } else if constexpr (NDIM == 2) {
            shell = std::numbers::pi * (r2 * r2 - r1 * r1);
        } else {
            shell = 2.0 * dr;
        }
        const double ideal = total_samples * npairs * shell / volume;
        const double g = (ideal > 0.0) ? total[k] / ideal : 0.0;
        file << std::format("{:>16.8e} {:>16.8e} {:>16.8e}\n",
                            Units::convertToUser("length", out_unit, r1 + 0.5 * dr), g, total[k]);
    }
}
