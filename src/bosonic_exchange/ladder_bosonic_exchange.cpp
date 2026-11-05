#include <array>
#include <fstream>
#include <cmath>

#include "bosonic_exchange/ladder_bosonic_exchange.h"
#include "simulation.h"
#include "winding.h"

LadderBosonicExchange::LadderBosonicExchange(const Simulation& _sim) : BosonicExchangeBase(_sim),
    recursion(nbosons),
    logS(nbosons), logB(std::max(nbosons - 1, 0)), logR(std::max(nbosons - 1, 0)), logL(std::max(nbosons - 2, 0)),
    AS(nbosons), AB(std::max(nbosons - 1, 0)), AR(std::max(nbosons - 1, 0)), AL(std::max(nbosons - 2, 0)),
    log_n_factorial(std::lgamma(nbosons + 1)) {
    evaluateKernels();
}

void LadderBosonicExchange::prepare() {
    evaluateKernels();
}

/**
 * Log kernel of the link from the last bead of a to the first bead of b. The link carries the
 * stiffness of the particle it LEAVES (a), exactly as in the quadratic class, so the softened block
 * transfers unchanged. With winding-sum springs the kernel is the image-summed weight mu(d) and
 * A is (k/2)<|d + wL|^2>; otherwise both reduce to (k/2) d^2.
 */
void LadderBosonicExchange::linkKernel(const dVec& x_first, const dVec& x_last, const int a, const int b,
                                       double& log_kernel, double& a_val) const {
    const double k = sim.linkStiffness(a);
    double d[NDIM];
    getBeadsSeparation(x_last, a, x_first, b, d);
    if (sim.winding_springs) {
        log_kernel = sim.linkLogWeight(d, k);
        a_val = sim.linkEnergyExpectation(d, k);
        return;
    }
    double d2 = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) d2 += d[axis] * d[axis];
    a_val = 0.5 * k * d2;
    log_kernel = -beta * a_val;
}

void LadderBosonicExchange::evaluateKernels() {
    dVec x_first_bead(nbosons);
    dVec x_last_bead(nbosons);
    assignFirstLast(x_first_bead, x_last_bead);

    const int n = nbosons;
    for (int a = 0; a < n; ++a) {
        linkKernel(x_first_bead, x_last_bead, a, a, logS[a], AS[a]);
    }
    for (int a = 0; a < n - 1; ++a) {
        linkKernel(x_first_bead, x_last_bead, a, a + 1, logB[a], AB[a]);
        linkKernel(x_first_bead, x_last_bead, a + 1, a, logR[a], AR[a]);
    }
    for (int a = 0; a < n - 2; ++a) {
        if (LadderRecursion::rungForward(a)) {
            linkKernel(x_first_bead, x_last_bead, a, a + 2, logL[a], AL[a]);
        } else {
            linkKernel(x_first_bead, x_last_bead, a + 2, a, logL[a], AL[a]);
        }
    }
    recursion.compute(logS, logB, logR, logL);

    if (!std::isfinite(recursion.logAlphaN())) {
        throw std::overflow_error("Invalid ladder exchange potential (non-finite ln alpha_N)");
    }
}

double LadderBosonicExchange::effectivePotential() {
    return -recursion.logAlphaN() / beta;
}

/**
 * Force on the last beads (rank P-1): every candidate link LEAVING l, weighted by its probability,
 * plus the interior link to the previous bead.
 */
void LadderBosonicExchange::springForceLastBead(dVec& f) {
    const int n = nbosons;
    for (int l = 0; l < n; ++l) {
        std::array<double, NDIM> sums = {};
        const int candidates[4] = {l, l + 1, l - 1, LadderRecursion::rungForward(l) ? l + 2 : l - 2};
        for (int next_l : candidates) {
            if (next_l < 0 || next_l >= n) continue;
            const double prob = recursion.connectionProbability(l, next_l);
            if (prob == 0.0) continue;
            double diff_next[NDIM];
            getBeadsSeparation(x, l, x_next, next_l, diff_next);
            if (sim.winding_springs) sim.linkMeanSeparation(diff_next, diff_next, sim.linkStiffness(l));
            for (int axis = 0; axis < NDIM; ++axis) {
                sums[axis] += prob * diff_next[axis] * sim.linkStiffness(l);
            }
        }

        double diff_prev[NDIM];
        getBeadsSeparation(x, l, x_prev, l, diff_prev);
        if (sim.winding_springs) sim.linkMeanSeparation(diff_prev, diff_prev);
        for (int axis = 0; axis < NDIM; ++axis) {
            sums[axis] += diff_prev[axis] * spring_constant;  // interior link
        }
        for (int axis = 0; axis < NDIM; ++axis) {
            f(l, axis) = sums[axis];
        }
    }
}

/**
 * Force on the first beads (rank 0): every candidate link ENTERING l (it carries the stiffness of
 * the particle it leaves), weighted by its probability, plus the interior link to the next bead.
 */
void LadderBosonicExchange::springForceFirstBead(dVec& f) {
    const int n = nbosons;
    for (int l = 0; l < n; ++l) {
        std::array<double, NDIM> sums = {};
        // incoming: from l, l-1 (B[l-1]), l+1 (R[l]), l-2 (rung L[l-2], forward since l-2 is even
        // when l is even), l+2 (rung L[l], backward since l is odd)
        const int candidates[4] = {l, l - 1, l + 1, (l % 2 == 0) ? l - 2 : l + 2};
        for (int prev_l : candidates) {
            if (prev_l < 0 || prev_l >= n) continue;
            const double prob = recursion.connectionProbability(prev_l, l);
            if (prob == 0.0) continue;
            double diff_prev[NDIM];
            getBeadsSeparation(x, l, x_prev, prev_l, diff_prev);
            if (sim.winding_springs) sim.linkMeanSeparation(diff_prev, diff_prev, sim.linkStiffness(prev_l));
            for (int axis = 0; axis < NDIM; ++axis) {
                sums[axis] += prob * diff_prev[axis] * sim.linkStiffness(prev_l);
            }
        }

        double diff_next[NDIM];
        getBeadsSeparation(x, l, x_next, l, diff_next);
        if (sim.winding_springs) sim.linkMeanSeparation(diff_next, diff_next);
        for (int axis = 0; axis < NDIM; ++axis) {
            sums[axis] += diff_next[axis] * spring_constant;  // interior link
        }
        for (int axis = 0; axis < NDIM; ++axis) {
            f(l, axis) = sums[axis];
        }
    }
}

/// Probability of the identity composition (all particles separate), including its 1/N!
double LadderBosonicExchange::getDistinctProbability() {
    return std::exp(recursion.logIdentity() - recursion.logAlphaN());
}

/// Weight of the single block [0, N-1] over alpha_N, i.e. N times the probability of an N-cycle
/// (the quadratic class returns the same quantity for its representative).
double LadderBosonicExchange::getLongestProbability() {
    return std::exp(recursion.logLongestBlock() - recursion.logAlphaN());
}

/**
 * d(beta V_B)/d beta at fixed coordinates = -sum_K p_K A_K: the estimator is linear in the links,
 * so the link probabilities of the reverse pass give it directly (no second recursion).
 */
double LadderBosonicExchange::primEstimator() {
    const double est = -recursion.linearExpectation(AS, AB, AR, AL);
#if IPI_CONVENTION
    return est / nbeads;
#else
    return est;
#endif
}

void LadderBosonicExchange::samplePermutation(std::vector<int>& perm, std::mt19937& gen) const {
    recursion.sample(perm, gen);
}

double LadderBosonicExchange::getConnectionProbability(int l, int u) const {
    return recursion.connectionProbability(l, u);
}

void LadderBosonicExchange::printBosonicDebug() {
    if (sim.this_bead == 0) {
        std::ofstream debug;
        debug.open(std::format("{}/bosonic_debug.log", Output::FOLDER_NAME), std::ios::out | std::ios::app);
        debug << "Step " << sim.getStep() << '\n';
        debug << "ln alpha_N = " << recursion.logAlphaN() << ", V_B = " << effectivePotential() << '\n';
        for (int m = 1; m <= nbosons; ++m) {
            debug << std::format("V[{}] = {}\n", m, -recursion.logAlpha(m) / beta);
        }
        debug << "----\nConnection probabilities (last bead of l -> first bead of u):\n";
        for (int l = 0; l < nbosons; ++l) {
            for (int u = std::max(0, l - 2); u <= std::min(nbosons - 1, l + 2); ++u) {
                const double p = recursion.connectionProbability(l, u);
                if (p > 0.0) debug << std::format("P[l={}, u={}] = {}\n", l, u, p);
            }
        }
        debug << "===========\n";
        debug.close();
    }
}
