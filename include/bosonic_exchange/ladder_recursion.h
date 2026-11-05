#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

/**
 * @class LadderRecursion
 * @brief Linear-time exact bosonic exchange on the "ladder" cycle representative.
 *
 * The quadratic Feldman-Hirshberg recursion sums over compositions of the labels 0..N-1 into blocks
 * of consecutive labels, with the block ending at (0-based) label v weighted by 1/(v+1), and
 * represents each block [u,v] by the cycle u -> u+1 -> ... -> v -> u. After integration over the
 * coordinates only the cycle TYPE of the representative matters (relabelling whole ring polymers
 * maps one representative onto another of the same type, and the composition weights sum to the
 * fraction of permutations of that type), so any representative of the same type gives the same
 * partition function and the same averages of every quantity invariant under simultaneous
 * relabelling of coordinates and permutation. This class represents the block [u,v] by the LADDER
 * cycle that climbs the even labels and descends the odd ones,
 *
 *     [0,5]: 0 -> 2 -> 4 -> 5 -> 3 -> 1 -> 0,     [1,6]: 1 -> 2 -> 4 -> 6 -> 5 -> 3 -> 1,
 *
 * whose links join labels at most two apart. Every block is then a product of per-link kernels
 * taken from a fixed set of 4N - 4 candidates (last bead of a -> first bead of b):
 *
 *     S[a]: a -> a         (N)
 *     B[a]: a -> a+1       (N-1)
 *     R[a]: a+1 -> a       (N-1)
 *     L[a]: the rung on the pair (a, a+2): a -> a+2 for a even, a+2 -> a for a odd   (N-2)
 *
 * with w(u,v) = c_u (prod_{i=u}^{v-2} L[i]) d_v, where c_u = B[u] (u odd) or R[u] (u even) and
 * d_v = R[v-1] (v even) or B[v-1] (v odd). Each candidate plays exactly one role, so the total
 * weight alpha_N is multilinear in the kernels and a forward pass (alpha, O) plus a backward pass
 * (alpha~, R) give alpha_N, every link probability p_K = K dalpha_N/dK / alpha_N, and an exact
 * conditional sampler, all in O(N) time and memory. Everything is in log space; every sum has two
 * terms. Derivation: docs/12_linear_ladder_exchange.md of the benchmark repository.
 *
 * The class knows nothing about coordinates: it takes log kernels (-beta E per link) and returns
 * log alpha_N (= -beta V), the link probabilities and samples of the ladder permutation.
 */
class LadderRecursion {
public:
    explicit LadderRecursion(int n_) : n(n_) {
        if (n < 1) throw std::invalid_argument("LadderRecursion needs at least one particle");
        lS.assign(n, 0.0); lB.assign(std::max(n - 1, 0), 0.0);
        lR.assign(std::max(n - 1, 0), 0.0); lL.assign(std::max(n - 2, 0), 0.0);
        A.assign(n + 1, NEG); O.assign(n + 1, NEG); AT.assign(n + 1, NEG); RR.assign(n, NEG);
        pS_.assign(n, 0.0); pB_.assign(std::max(n - 1, 0), 0.0);
        pR_.assign(std::max(n - 1, 0), 0.0); pL_.assign(std::max(n - 2, 0), 0.0);
    }

    [[nodiscard]] int size() const { return n; }

    /**
     * Runs both passes for the given log kernels (sizes N, N-1, N-1, N-2).
     */
    void compute(const std::vector<double>& logS, const std::vector<double>& logB,
                 const std::vector<double>& logR, const std::vector<double>& logL) {
        if (static_cast<int>(logS.size()) != n || static_cast<int>(logB.size()) != n - 1 ||
            static_cast<int>(logR.size()) != n - 1 || static_cast<int>(logL.size()) != std::max(n - 2, 0)) {
            throw std::invalid_argument("LadderRecursion::compute: kernel arrays of the wrong size");
        }
        lS = logS; lB = logB; lR = logR; lL = logL;

        // Forward pass. A[m]: compositions of the prefix of m labels (0..m-1). O[m] (m >= 2): open
        // blocks [u, m-1] with u <= m-2, extended through the rung (m-3, m-1) but not yet closed.
        A[0] = 0.0;
        A[1] = lS[0];
        for (int m = 2; m <= n; ++m) {
            O[m] = (m == 2) ? A[0] + logc(0) : lse(O[m - 1] + lL[m - 3], A[m - 2] + logc(m - 2));
            A[m] = lse(A[m - 1] + lS[m - 1], O[m] + logd(m - 1)) - std::log(static_cast<double>(m));
        }

        // Backward pass. AT[m]: compositions of the suffix m..n-1 (block ending at label v carries
        // 1/(v+1)). RR[u] = sum_{v >= u+1} (prod_{i=u}^{v-2} L[i]) d_v AT[v+1] / (v+1).
        AT[n] = 0.0;
        for (int u = n - 1; u >= 0; --u) {
            if (u <= n - 2) {
                const double close_next = logd(u + 1) + AT[u + 2] - std::log(static_cast<double>(u + 2));
                RR[u] = (u <= n - 3) ? lse(close_next, lL[u] + RR[u + 1]) : close_next;
            } else {
                RR[u] = NEG;
            }
            const double single = lS[u] + AT[u + 1] - std::log(static_cast<double>(u + 1));
            AT[u] = (u <= n - 2) ? lse(single, logc(u) + RR[u]) : single;
        }

        // Link probabilities (multilinearity: p_K = K dalpha/dK / alpha).
        const double lz = A[n];
        for (int v = 0; v < n; ++v) {
            pS_[v] = std::exp(A[v] + lS[v] + AT[v + 1] - std::log(static_cast<double>(v + 1)) - lz);
        }
        for (int a = 0; a < n - 1; ++a) {
            // c at u = a  (block starting at a, a <= n-2) and d at v = a+1 (block ending at a+1)
            const double pc = std::exp(A[a] + logc(a) + RR[a] - lz);
            const double pd = std::exp(O[a + 2] + logd(a + 1) + AT[a + 2] - std::log(static_cast<double>(a + 2)) - lz);
            if (a % 2 == 1) { pB_[a] = pc; pR_[a] = pd; } else { pR_[a] = pc; pB_[a] = pd; }
        }
        for (int i = 0; i < n - 2; ++i) {
            pL_[i] = std::exp(O[i + 2] + lL[i] + RR[i + 1] - lz);
        }
    }

    /// ln alpha_N = -beta V_B
    [[nodiscard]] double logAlphaN() const { return A[n]; }
    /// ln alpha_m for the prefix of m labels
    [[nodiscard]] double logAlpha(int m) const { return A[m]; }
    /// ln alpha~_0, which must equal ln alpha_N (consistency of the two passes)
    [[nodiscard]] double logAlphaTilde0() const { return AT[0]; }

    [[nodiscard]] const std::vector<double>& pS() const { return pS_; }
    [[nodiscard]] const std::vector<double>& pB() const { return pB_; }
    [[nodiscard]] const std::vector<double>& pR() const { return pR_; }
    [[nodiscard]] const std::vector<double>& pL() const { return pL_; }

    /// Label joined by the rung L[a]: a+2 for a even (a -> a+2), and the rung a+2 -> a for a odd.
    static bool rungForward(int a) { return a % 2 == 0; }

    /// Probability that the last bead of l connects to the first bead of u (zero off the ladder).
    [[nodiscard]] double connectionProbability(int l, int u) const {
        if (l < 0 || u < 0 || l >= n || u >= n) return 0.0;
        if (u == l) return pS_[l];
        if (u == l + 1) return pB_[l];
        if (u == l - 1) return pR_[u];
        if (u == l + 2 && rungForward(l)) return pL_[l];
        if (u == l - 2 && !rungForward(u)) return pL_[u];
        return 0.0;
    }

    /// sum_K p_K A_K for per-link quantities A (same layout as the kernels).
    [[nodiscard]] double linearExpectation(const std::vector<double>& AS, const std::vector<double>& AB,
                                           const std::vector<double>& AR, const std::vector<double>& AL) const {
        double sum = 0.0;
        for (int v = 0; v < n; ++v) sum += pS_[v] * AS[v];
        for (int a = 0; a < n - 1; ++a) sum += pB_[a] * AB[a] + pR_[a] * AR[a];
        for (int i = 0; i < n - 2; ++i) sum += pL_[i] * AL[i];
        return sum;
    }

    /// ln of the weight of the single block [0, n-1] (the N-cycle on the ladder)
    [[nodiscard]] double logLongestBlock() const {
        if (n == 1) return lS[0];
        double val = logc(0) + logd(n - 1);
        for (int i = 0; i <= n - 3; ++i) val += lL[i];
        return val;
    }

    /// ln of the weight of the identity composition, including its 1/N! coefficient
    [[nodiscard]] double logIdentity() const {
        double val = 0.0;
        for (int v = 0; v < n; ++v) val += lS[v] - std::log(static_cast<double>(v + 1));
        return val;
    }

    /**
     * Writes the ladder cycle of the block [u, v] into perm (perm[l] = successor of l).
     */
    static void blockCycle(int u, int v, std::vector<int>& perm) {
        if (u == v) { perm[u] = u; return; }
        std::vector<int> seq;
        seq.reserve(v - u + 1);
        if (u % 2 == 1) seq.push_back(u);
        for (int e = (u % 2 == 0) ? u : u + 1; e <= v; e += 2) seq.push_back(e);
        const int top_odd = (v % 2 == 1) ? v : v - 1;
        const int low_odd = (u % 2 == 0) ? u + 1 : u + 2;
        for (int o = top_odd; o >= low_odd; o -= 2) seq.push_back(o);
        for (size_t i = 0; i + 1 < seq.size(); ++i) perm[seq[i]] = seq[i + 1];
        perm[seq.back()] = seq.front();
    }

    /**
     * Draws a composition from the exact conditional distribution (term weight over alpha_N) and
     * returns the corresponding ladder permutation. O(N) per sample: from the top, the block ending
     * at the current label is a singleton with probability alpha_{m-1} s / (m alpha_m), otherwise
     * an open block whose start is found by walking the O recursion down one label at a time
     * ("extended" with probability O_{m-1} L_{m-3} / O_m, "started at m-2" otherwise).
     */
    template <class Generator>
    void sample(std::vector<int>& perm, Generator& gen) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        perm.assign(n, -1);
        int m = n;
        while (m > 0) {
            const int v = m - 1;
            const double p_single = (m == 1) ? 1.0
                : std::exp(A[m - 1] + lS[v] - std::log(static_cast<double>(m)) - A[m]);
            if (uniform(gen) < p_single) {
                perm[v] = v;
                m -= 1;
                continue;
            }
            int mm = m;
            int u = 0;
            while (true) {
                if (mm == 2) { u = 0; break; }
                const double p_ext = std::exp(O[mm - 1] + lL[mm - 3] - O[mm]);
                if (uniform(gen) < p_ext) {
                    mm -= 1;
                } else {
                    u = mm - 2;
                    break;
                }
            }
            blockCycle(u, v, perm);
            m = u;
        }
    }

private:
    static constexpr double NEG = -std::numeric_limits<double>::infinity();

    static double lse(double a, double b) {
        if (a == NEG) return b;
        if (b == NEG) return a;
        const double hi = std::max(a, b);
        return hi + std::log1p(std::exp(std::min(a, b) - hi));
    }

    /// Start link of a block beginning at u (u <= n-2): B[u] for u odd, R[u] for u even.
    [[nodiscard]] double logc(int u) const { return (u % 2 == 1) ? lB[u] : lR[u]; }
    /// Top link of a block ending at v (v >= 1): R[v-1] for v even, B[v-1] for v odd.
    [[nodiscard]] double logd(int v) const { return (v % 2 == 0) ? lR[v - 1] : lB[v - 1]; }

    int n;
    std::vector<double> lS, lB, lR, lL;
    std::vector<double> A, O, AT, RR;
    std::vector<double> pS_, pB_, pR_, pL_;
};
