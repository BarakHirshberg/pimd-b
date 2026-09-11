#pragma once

#include <cmath>
#include <random>

/**
 * @class WindingProbability
 * @brief One Cartesian component of a ring-polymer link in a periodic box with the winding sum of
 * Higer, Feldman & Hirshberg, J. Chem. Phys. 163, 024101 (2025).
 *
 * The rigorous periodic weight of a link with minimum-image separation d is
 *   mu(d) = sum_{w=-max_wind}^{max_wind} exp(-beta_half_k (d + w L)^2),   beta_half_k = beta_P k / 2,
 * which defines a discrete probability p_w over the winding number w of that link. This class evaluates
 * log mu, the expectation values <w> and <(d + wL)^2> needed for forces and the kinetic estimator, and
 * samples w for the exact winding-number estimator. Adapted from the `pbc` branch of higj/pimd-b.
 */
class WindingProbability {
public:
    WindingProbability(double diff_, int max_wind_, double beta_half_k_, double size_) :
        diff(diff_), max_wind(max_wind_), beta_half_k(beta_half_k_), size(size_) {
        // Shift the exponents by the smallest squared image separation for numerical stability
        shift = diff * diff;
        for (int w = 1; w <= max_wind; ++w) {
            shift = std::min(shift, std::min(sq(diff + w * size), sq(diff - w * size)));
        }
        denominator = 0.0;
        for (int w = -max_wind; w <= max_wind; ++w) {
            denominator += unnormalized(w);
        }
    }

    /// ln mu(d)
    [[nodiscard]] double logWeight() const { return std::log(denominator) - beta_half_k * shift; }

    /// p_w
    [[nodiscard]] double probability(int w) const { return unnormalized(w) / denominator; }

    /// <w>
    [[nodiscard]] double expectation() const {
        double mean = 0.0;
        for (int w = 1; w <= max_wind; ++w) {
            mean += w * (probability(w) - probability(-w));
        }
        return mean;
    }

    /// <(d + wL)^2>
    [[nodiscard]] double diffSquaredExpectation() const {
        double mean = 0.0;
        for (int w = -max_wind; w <= max_wind; ++w) {
            mean += sq(diff + w * size) * probability(w);
        }
        return mean;
    }

    /// Draw a winding number from p_w
    template <class Generator>
    int sample(Generator& gen) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double u = uniform(gen);
        double cumulative = 0.0;
        for (int w = -max_wind; w <= max_wind; ++w) {
            cumulative += probability(w);
            if (u < cumulative) return w;
        }
        return max_wind;
    }

private:
    double diff, shift, denominator;
    int max_wind;
    double beta_half_k, size;

    static double sq(double x) { return x * x; }
    [[nodiscard]] double unnormalized(int w) const { return std::exp(-beta_half_k * (sq(diff + w * size) - shift)); }
};
