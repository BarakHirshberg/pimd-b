#pragma once

#include <random>
#include <vector>

#include "observables/observable.h"

class Simulation; // Forward declaration

/* -------------------------------- */

/**
 * @class PermutationObservable
 * @brief Cycle-length statistics and winding number from permutations sampled exactly from the
 * bosonic effective potential (see BosonicExchangeBase::samplePermutation).
 *
 * Per measurement, n_perm_samples permutations are drawn on the first time slice. Quantities:
 *  - n_cycles:  number of permutation cycles
 *  - cycle_max: length of the longest cycle
 *  - p_exch:    fraction of particles in cycles of length >= 2
 *  - w2:        squared total winding vector (length^2, output unit^2) [only with winding = on]
 *  - p_exch_geom, w2_geom, rho_s_geom: the same quantities from the APPROXIMATE geometric permutation of
 *    Myung, Hirshberg & Parrinello, PRL 128, 045301 (2022), SI Alg. 1 (last bead of each particle joined to
 *    the nearest first bead), for comparison only [only with winding = on]
 *  - rho_s:     superfluid fraction estimator <W^2>/(2 lambda beta N d), lambda = hbar^2/(2m)
 * A cumulative histogram of cycle lengths is written to output/cycles.dat every sfreq steps.
 *
 * Winding: W = sum over all springs of the minimum-image displacement, i.e. the interior
 * displacements of every particle (reduced over ranks) plus the exterior links r_l^P -> r_{perm[l]}^1.
 * For a closed configuration W is an integer multiple of the box vector up to round-off.
 */
class PermutationObservable : public Observable {
public:
    PermutationObservable(const Simulation& _sim, int _freq, const std::string& _out_unit, bool _winding);
    ~PermutationObservable() override;

    void calculate() override;

private:
    void writeHistogram();

    bool winding;
    std::vector<int> gperm;  // Geometric (nearest-neighbour) permutation, Myung et al. PRL 2022 SI Alg. 1

    void geometricPermutation();
    int nsamples;
    std::mt19937 gen;
    std::vector<int> perm;
    std::vector<long> hist;           // Cumulative cycle-length histogram (index k-1 = cycles of length k)
    long total_samples;
    std::string hist_filename;
};
