#pragma once

#include "common.h"
#include "bosonic_exchange/bosonic_exchange_base.h"
#include "bosonic_exchange/ladder_recursion.h"

/**
 * @class LadderBosonicExchange
 * @brief Exact bosonic exchange in O(N) time and memory on the ladder cycle representative.
 *
 * Same interface and same conventions as the quadratic class (BosonicExchange): every exterior
 * link carries the stiffness of the particle it LEAVES (softened block), winding-sum springs use
 * the image-summed kernel, and the primitive estimator is -sum_K p_K A_K with A_K the spring-energy
 * expectation of link K. Only 4N - 4 candidate links exist; see LadderRecursion for the
 * representative and the derivation. Selected with [simulation] bosonic_algorithm = ladder.
 */
class LadderBosonicExchange final : public BosonicExchangeBase {
public:
    LadderBosonicExchange(const Simulation& _sim);
    ~LadderBosonicExchange() override = default;

    double effectivePotential() override;
    void prepare() override;
    double primEstimator() override;

    double getDistinctProbability() override;
    double getLongestProbability() override;

    void printBosonicDebug() override;

    void samplePermutation(std::vector<int>& perm, std::mt19937& gen) const override;
    double getConnectionProbability(int l, int u) const override;

protected:
    void springForceFirstBead(dVec& f) override;
    void springForceLastBead(dVec& f) override;

private:
    void evaluateKernels();
    /// Log kernel (-beta E) and spring-energy expectation of the link last bead of a -> first bead of b
    void linkKernel(const dVec& x_first, const dVec& x_last, int a, int b, double& log_kernel, double& a_val) const;

    LadderRecursion recursion;
    std::vector<double> logS, logB, logR, logL;  // -beta E of the 4N - 4 candidate links
    std::vector<double> AS, AB, AR, AL;          // spring-energy expectation per link (A = E without winding sums)
    double log_n_factorial;
};
