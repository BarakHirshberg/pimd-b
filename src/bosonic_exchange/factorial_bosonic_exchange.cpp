#include <array>
#include <random>
#include <numeric>
#include <algorithm>

#include "bosonic_exchange/factorial_bosonic_exchange.h"
#include "simulation.h"

FactorialBosonicExchange::FactorialBosonicExchange(const Simulation& _sim) : BosonicExchangeBase(_sim), labels(_sim.natoms) {
    if (_sim.winding_springs) {
        throw std::invalid_argument("winding_springs is implemented for the quadratic bosonic algorithm only!");
    }
    // Fill the labels array with numbers from 0 to nbosons-1
    std::iota(labels.begin(), labels.end(), 0);

    // For numerical stability
    e_shift = getMinExteriorSpringEnergy();
}

/**
 * @brief Recalculates the smallest exterior spring energy after each coordinate update.
 */
void FactorialBosonicExchange::prepare() {
    e_shift = getMinExteriorSpringEnergy();
}

/**
 * Identifies the particle to which the neighboring P bead, connected to the first bead of ptcl_idx, belongs.
 * In the context of Cauchy's two-line notation, it retrieves the number from the upper line, placed above ptcl_idx.
 * Equivalently, it is the position of ptcl_idx in the bottom line, assuming zero-based indexing.
 * This works because the upper line can be viewed as the list of particles in the last imaginary time slice and
 * the bottom line as the list of their respective neighbors in the first imaginary time slice.
 *
 * @param ptcl_idx Index of the particle associated with the first bead.
 * @return Index of the particle associated with the neighboring P bead.
 */
int FactorialBosonicExchange::firstBeadNeighbor(int ptcl_idx) const {
    return std::distance(labels.begin(), std::ranges::find(labels, ptcl_idx));
}

/**
 * Identifies the particle to which the neighboring 1 bead, connected to the last bead of ptcl_idx, belongs.
 * In the context of Cauchy's two-line notation, it retrieves the number from the bottom line, placed below ptcl_idx.
 * Equivalently, it is the particle label in the bottom line at position ptcl_idx, assuming zero-based indexing.
 * This works because the upper line can be viewed as the list of particles in the last imaginary time slice and
 * the bottom line as the list of their respective neighbors in the first imaginary time slice.
 *
 * @param ptcl_idx Index of the particle associated with the last bead.
 * @return Index of the particle associated with the neighboring 1 bead.
 */
int FactorialBosonicExchange::lastBeadNeighbor(int ptcl_idx) const {
    return labels[ptcl_idx];
}

/**
 * Calculates the smallest exterior spring energy that a permutation can yield in a given time-step. This energy is then used to shift
 * the spring energies in the Boltzmann weights to avoid numerical instabilities, which can be especially prominent
 * at high Trotter numbers.
 *
 * @return The smallest exterior spring energy contribution due to a permutation.
 */
double FactorialBosonicExchange::getMinExteriorSpringEnergy() {
    dVec x_first_bead(nbosons);
    dVec x_last_bead(nbosons);

    assignFirstLast(x_first_bead, x_last_bead);

    double min_delta = std::numeric_limits<double>::max();

    // Iterate over all permutations
    do {
        double diff2 = 0.0;

        for (int l = 0; l < nbosons; ++l) {
            std::array<double, NDIM> sums = {};

            // First bead of some particle (depending on the permutation) minus last bead of the l-th particle
            diff2 += getBeadsSeparationSquared(x_first_bead, l, x_last_bead, lastBeadNeighbor(l));
        }

        // Compare the current total exterior spring energy with the minimum total exterior spring energy found so far
        min_delta = std::min(min_delta, 0.5 * spring_constant * diff2);
    } while (std::ranges::next_permutation(labels).found);

    return min_delta;
}

/**
 * Calculates the effective bosonic spring potential. This is V_eff from the exp(-beta * V_eff) in the
 * partition function. In contrast to the classical spring energy of interior beads,
 * this contribution is no longer harmonic, but is a sum of Boltzmann weights due to all permutations.
 *
 * @return Effective bosonic exchange potential.
 */
double FactorialBosonicExchange::effectivePotential() {
    dVec x_first_bead(nbosons);
    dVec x_last_bead(nbosons);

    assignFirstLast(x_first_bead, x_last_bead);

    long permutation_counter = 0;
    double weights_sum = 0.0;

    // Iterate over all permutations and calculate the weights
    // associated with the exterior beads.
    do {
        permutation_counter++;
        double diff2 = 0.0;

        for (int ptcl_idx = 0; ptcl_idx < nbosons; ++ptcl_idx) {
            diff2 += getBeadsSeparationSquared(x_first_bead, ptcl_idx, x_last_bead, lastBeadNeighbor(ptcl_idx));
        }

        weights_sum += exp(-sim.beta_half_k * diff2);
    } while (std::ranges::next_permutation(labels).found);

    return (-1.0 / beta) * log(weights_sum / permutation_counter);
}

/**
 * Evaluates the bosonic spring forces acting on the particles at the last imaginary time slice.
 *
 * @param[out] f Spring forces acting on the particles at time-slice P.
 */
void FactorialBosonicExchange::springForceLastBead(dVec& f) {
    const dVec x_first_bead = x_next;
    const dVec x_last_bead = x;

    dVec temp_force(nbosons);
    double denom_weight = 0.0;

    do {
        double weight = 0.0;

        for (int l = 0; l < nbosons; ++l) {
            std::array<double, NDIM> sums = {};

            double diff_next[NDIM];

            // First bead (1) of some particle (depending on the permutation) minus last bead (P) of the l-th particle
            getBeadsSeparation(x_last_bead, l, x_first_bead, lastBeadNeighbor(l), diff_next);

            // Coordinate differences corresponding to exterior beads
            for (int axis = 0; axis < NDIM; ++axis) {
                weight += diff_next[axis] * diff_next[axis];
            }

            double diff_prev[NDIM];

            // Previous bead (P-1) of the l-th particle minus last bead (P) of the l-th particle
            getBeadsSeparation(x_last_bead, l, x_prev, l, diff_prev);

            for (int axis = 0; axis < NDIM; ++axis) {
                sums[axis] += diff_prev[axis] + diff_next[axis];
                temp_force(l, axis) = sums[axis] * spring_constant;
            }
        }

        weight = exp(-beta * (0.5 * spring_constant * weight - e_shift));

        for (int l = 0; l < nbosons; ++l) {
            for (int axis = 0; axis < NDIM; ++axis) {
                f(l, axis) += weight * temp_force(l, axis);
            }
        }

        denom_weight += weight;
    } while (std::ranges::next_permutation(labels).found);

    // Normalize the forces by the sum of Boltzmann contributions due to all the permutations
    for (int l = 0; l < nbosons; ++l) {
        for (int axis = 0; axis < NDIM; ++axis) {
            f(l, axis) = f(l, axis) / denom_weight;
        }
    }
}

/**
 * Evaluates the bosonic spring forces acting on the particles at the first imaginary time slice.
 *
 * @param[out] f Spring forces acting on the particles at time-slice 1.
 */
void FactorialBosonicExchange::springForceFirstBead(dVec& f) {
    const dVec x_first_bead = x;
    const dVec x_last_bead = x_prev;

    dVec temp_force(nbosons);
    double denom_weight = 0.0;

    do {
        double weight = 0.0;

        for (int l = 0; l < nbosons; ++l) {
            std::array<double, NDIM> sums = {};

            double diff_prev[NDIM];

            // Last bead (P) of some particle (depending on the permutation) minus first bead (1) of the l-th particle
            getBeadsSeparation(x_first_bead, l, x_last_bead, firstBeadNeighbor(l), diff_prev);

            // Coordinate differences corresponding to exterior beads
            for (int axis = 0; axis < NDIM; ++axis) {
                weight += diff_prev[axis] * diff_prev[axis];
            }

            double diff_next[NDIM];

            // Next bead (2) of the l-th particle minus first bead (1) of the l-th particle
            getBeadsSeparation(x_first_bead, l, x_next, l, diff_next);

            for (int axis = 0; axis < NDIM; ++axis) {
                sums[axis] += diff_prev[axis] + diff_next[axis];
                temp_force(l, axis) = sums[axis] * spring_constant;
            }
        }

        weight = exp(-beta * (0.5 * spring_constant * weight - e_shift));

        for (int l = 0; l < nbosons; ++l) {
            for (int axis = 0; axis < NDIM; ++axis) {
                f(l, axis) += weight * temp_force(l, axis);
            }
        }

        denom_weight += weight;
    } while (std::ranges::next_permutation(labels).found);

    // Normalize the forces by the sum of Boltzmann contributions due to all the permutations
    for (int l = 0; l < nbosons; ++l) {
        for (int axis = 0; axis < NDIM; ++axis) {
            f(l, axis) = f(l, axis) / denom_weight;
        }
    }
}

/**
 * Evaluate the probability of the configuration where all the particles are separate.
 *
 * @return Probability of a configuration corresponding to the identity permutation.
 */
double FactorialBosonicExchange::getDistinctProbability() {
    /// @todo Currently not implemented
    return 0.0;
}

/**
 * Evaluate the probability of a configuration where all the particles are connected,
 * divided by 1/N. Notice that there are (N-1)! permutations of this topology
 * (all represented by the cycle 0,1,...,N-1,0); this cancels the division by 1/N.
 *
 * @return Probability of a configuration where all the particles are connected.
 */
double FactorialBosonicExchange::getLongestProbability() {
    /// @todo Currently not implemented
    return 0.0;
}

/**
 * Calculates the contribution of the exterior beads to the primitive kinetic energy estimator.
 *
 * @return Weighted average of exterior spring energies over all permutations. 
 */
double FactorialBosonicExchange::primEstimator() {
    dVec x_first_bead(nbosons);
    dVec x_last_bead(nbosons);

    assignFirstLast(x_first_bead, x_last_bead);

    double numerator = 0.0;
    double denom_weight = 0.0;

    do {
        double weight = 0.0;

        for (int l = 0; l < nbosons; ++l) {
            // Last bead of some particle (depending on the permutation) minus first bead of the l-th particle
            weight += getBeadsSeparationSquared(x_first_bead, l, x_last_bead, firstBeadNeighbor(l));
        }

        double delta_spring_energy = 0.5 * spring_constant * weight;
        weight = exp(-beta * (delta_spring_energy - e_shift));

        // Delta(E^sigma) is ready only at this point. We multiply Delta(E^sigma) by the weight
        // and add the result to the numerator (and also update the denominator by adding the 
        // weight of the current permutation).
        numerator += delta_spring_energy * weight;
        denom_weight += weight;
    } while (std::ranges::next_permutation(labels).found);

    double bosonic_spring_energy = numerator / denom_weight;

#if IPI_CONVENTION
    bosonic_spring_energy /= nbeads;
#endif

    return (-1.0) * bosonic_spring_energy;
}

/**
 * Exterior spring energy of a given permutation (last bead of l connected to first bead of
 * permutation[l]), used by the enumeration-based estimators.
 */
double FactorialBosonicExchange::exteriorSpringEnergy(const std::vector<int>& permutation) const {
    dVec x_first_bead(nbosons);
    dVec x_last_bead(nbosons);
    assignFirstLast(x_first_bead, x_last_bead);

    double diff2 = 0.0;
    for (int l = 0; l < nbosons; ++l) {
        diff2 += getBeadsSeparationSquared(x_last_bead, l, x_first_bead, permutation[l]);
    }
    return 0.5 * spring_constant * diff2;
}

/**
 * Samples a permutation with probability proportional to exp(-beta*E^sigma) by enumerating all N!
 * permutations (label independent; intended for N <= 7 as a reference for the quadratic algorithm).
 */
void FactorialBosonicExchange::samplePermutation(std::vector<int>& perm, std::mt19937& gen) const {
    std::vector<int> permutation(nbosons);
    std::iota(permutation.begin(), permutation.end(), 0);

    std::vector<std::vector<int>> all_perms;
    std::vector<double> weights;
    do {
        all_perms.push_back(permutation);
        weights.push_back(exp(-beta * (exteriorSpringEnergy(permutation) - e_shift)));
    } while (std::ranges::next_permutation(permutation).found);

    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    perm = all_perms[pick(gen)];
}

/**
 * Probability that the last bead of particle l is connected to the first bead of particle u,
 * obtained by enumerating all permutations.
 */
double FactorialBosonicExchange::getConnectionProbability(int l, int u) const {
    std::vector<int> permutation(nbosons);
    std::iota(permutation.begin(), permutation.end(), 0);

    double numerator = 0.0;
    double denominator = 0.0;
    do {
        const double weight = exp(-beta * (exteriorSpringEnergy(permutation) - e_shift));
        denominator += weight;
        if (permutation[l] == u) {
            numerator += weight;
        }
    } while (std::ranges::next_permutation(permutation).found);

    return numerator / denominator;
}

void FactorialBosonicExchange::printBosonicDebug() {
    /// @todo Currently not implemented
    return;
}