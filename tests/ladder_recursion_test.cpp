// Unit tests of LadderRecursion against brute force. Build with -DBUILD_LADDER_TESTS=ON.
//   1. ln alpha_N against the explicit sum over compositions (ladder cycles, 1/(v+1) weights) for
//      N = 1..7 with random kernels; forward and backward passes agree (alpha~_0 = alpha_N).
//   2. Link probabilities against central differences of ln alpha_N in each log kernel.
//   3. Each particle's outgoing and incoming probabilities sum to one.
//   4. With all kernels equal to one the sampled cycle-type frequencies equal |C_lambda| / N!, and
//      with random kernels the sampled composition frequencies equal their exact weights.
#include <cstdio>
#include <map>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

#include "bosonic_exchange/ladder_recursion.h"

namespace {

struct Kernels {
    std::vector<double> S, B, R, L;
};

Kernels randomKernels(int n, std::mt19937& gen, double spread) {
    std::normal_distribution<double> g(0.0, spread);
    Kernels k;
    k.S.resize(n); k.B.resize(std::max(n - 1, 0)); k.R.resize(std::max(n - 1, 0)); k.L.resize(std::max(n - 2, 0));
    for (auto& x : k.S) x = g(gen);
    for (auto& x : k.B) x = g(gen);
    for (auto& x : k.R) x = g(gen);
    for (auto& x : k.L) x = g(gen);
    return k;
}

// log kernel of the link a -> b, or throws if it is not a ladder candidate
double linkLog(const Kernels& k, int a, int b) {
    if (b == a) return k.S[a];
    if (b == a + 1) return k.B[a];
    if (b == a - 1) return k.R[b];
    if (b == a + 2 && a % 2 == 0) return k.L[a];
    if (b == a - 2 && b % 2 == 1) return k.L[b];
    throw std::runtime_error("link " + std::to_string(a) + " -> " + std::to_string(b) + " is not a ladder candidate");
}

double logWeightOfPerm(const Kernels& k, const std::vector<int>& perm) {
    double s = 0.0;
    for (int a = 0; a < static_cast<int>(perm.size()); ++a) s += linkLog(k, a, perm[a]);
    return s;
}

double lse(const std::vector<double>& v) {
    const double hi = *std::max_element(v.begin(), v.end());
    double s = 0.0;
    for (double x : v) s += std::exp(x - hi);
    return hi + std::log(s);
}

// Enumerate compositions of n labels; call f(blocks) with blocks = list of (u, v).
void forEachComposition(int n, const std::function<void(const std::vector<std::pair<int, int>>&)>& f) {
    std::vector<std::pair<int, int>> blocks;
    std::function<void(int)> rec = [&](int start) {
        if (start == n) { f(blocks); return; }
        for (int v = start; v < n; ++v) {
            blocks.emplace_back(start, v);
            rec(v + 1);
            blocks.pop_back();
        }
    };
    rec(0);
}

std::string compositionKey(const std::vector<std::pair<int, int>>& blocks) {
    std::string key;
    for (auto [u, v] : blocks) key += std::to_string(u) + "-" + std::to_string(v) + ",";
    return key;
}

std::string cycleTypeKey(const std::vector<int>& perm) {
    const int n = static_cast<int>(perm.size());
    std::vector<int> seen(n, 0), lengths;
    for (int s = 0; s < n; ++s) {
        if (seen[s]) continue;
        int len = 0;
        for (int l = s; !seen[l]; l = perm[l]) { seen[l] = 1; ++len; }
        lengths.push_back(len);
    }
    std::sort(lengths.begin(), lengths.end());
    std::string key;
    for (int l : lengths) key += std::to_string(l) + ",";
    return key;
}

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

}  // namespace

int main() {
    std::mt19937 gen(20261105);

    // 1 + 2 + 3: brute force, finite differences, sum rules
    for (int n = 1; n <= 7; ++n) {
        for (int trial = 0; trial < 3; ++trial) {
            Kernels k = randomKernels(n, gen, 1.5);
            LadderRecursion lad(n);
            lad.compute(k.S, k.B, k.R, k.L);

            std::vector<double> terms;
            std::map<std::string, double> exact;  // composition -> log weight
            std::vector<int> perm(n);
            forEachComposition(n, [&](const std::vector<std::pair<int, int>>& blocks) {
                double lw = 0.0;
                for (auto [u, v] : blocks) {
                    LadderRecursion::blockCycle(u, v, perm);
                    lw -= std::log(static_cast<double>(v + 1));
                }
                // every label must be covered exactly once by the blocks' cycles
                lw += logWeightOfPerm(k, perm);
                terms.push_back(lw);
                exact[compositionKey(blocks)] = lw;
            });
            const double brute = lse(terms);
            check(std::abs(brute - lad.logAlphaN()) < 1e-10,
                  "brute force vs recursion, N = " + std::to_string(n) + ": " + std::to_string(brute) + " vs " + std::to_string(lad.logAlphaN()));
            check(std::abs(lad.logAlphaTilde0() - lad.logAlphaN()) < 1e-10,
                  "backward pass, N = " + std::to_string(n));

            // finite differences: d ln alpha / d logK = p_K
            const double h = 1e-5;
            auto fd = [&](std::vector<double>& arr, int i) {
                const double x0 = arr[i];
                LadderRecursion t(n);
                arr[i] = x0 + h; t.compute(k.S, k.B, k.R, k.L); const double up = t.logAlphaN();
                arr[i] = x0 - h; t.compute(k.S, k.B, k.R, k.L); const double dn = t.logAlphaN();
                arr[i] = x0;
                return (up - dn) / (2 * h);
            };
            for (int i = 0; i < n; ++i)
                check(std::abs(fd(k.S, i) - lad.pS()[i]) < 1e-7, "pS FD, N = " + std::to_string(n) + " i = " + std::to_string(i));
            for (int i = 0; i < n - 1; ++i) {
                check(std::abs(fd(k.B, i) - lad.pB()[i]) < 1e-7, "pB FD, N = " + std::to_string(n) + " i = " + std::to_string(i));
                check(std::abs(fd(k.R, i) - lad.pR()[i]) < 1e-7, "pR FD, N = " + std::to_string(n) + " i = " + std::to_string(i));
            }
            for (int i = 0; i < n - 2; ++i)
                check(std::abs(fd(k.L, i) - lad.pL()[i]) < 1e-7, "pL FD, N = " + std::to_string(n) + " i = " + std::to_string(i));

            // sum rules
            for (int l = 0; l < n; ++l) {
                double out = 0.0, in = 0.0;
                for (int u = 0; u < n; ++u) { out += lad.connectionProbability(l, u); in += lad.connectionProbability(u, l); }
                check(std::abs(out - 1.0) < 1e-10, "outgoing sum, N = " + std::to_string(n) + " l = " + std::to_string(l));
                check(std::abs(in - 1.0) < 1e-10, "incoming sum, N = " + std::to_string(n) + " l = " + std::to_string(l));
            }

            // prob of the identity and of the N-cycle block
            check(std::abs(lad.logIdentity() - exact[compositionKey([&] {
                      std::vector<std::pair<int, int>> b; for (int i = 0; i < n; ++i) b.emplace_back(i, i); return b; }())]) < 1e-12,
                  "identity weight, N = " + std::to_string(n));
            check(std::abs(lad.logLongestBlock() - std::log(static_cast<double>(n)) - exact["0-" + std::to_string(n - 1) + ","]) < 1e-12,
                  "longest block weight, N = " + std::to_string(n));

            // 4b: sampler composition frequencies vs exact weights (N <= 5 to keep the table small)
            if (n <= 5 && trial == 0) {
                const int nsamp = 400000;
                std::map<std::string, int> counts;
                for (int s = 0; s < nsamp; ++s) {
                    lad.sample(perm, gen);
                    // recover the composition from the permutation: blocks are the cycles, each a
                    // set of consecutive labels
                    std::vector<int> seen(n, 0);
                    std::vector<std::pair<int, int>> blocks;
                    for (int st = 0; st < n; ++st) {
                        if (seen[st]) continue;
                        int lo = st, hi = st;
                        for (int l = st; !seen[l]; l = perm[l]) { seen[l] = 1; lo = std::min(lo, l); hi = std::max(hi, l); }
                        blocks.emplace_back(lo, hi);
                    }
                    std::sort(blocks.begin(), blocks.end());
                    // the permutation must be exactly the ladder cycle of those blocks
                    std::vector<int> rebuilt(n);
                    for (auto [u, v] : blocks) LadderRecursion::blockCycle(u, v, rebuilt);
                    check(rebuilt == perm, "sampled permutation is a ladder cycle, N = " + std::to_string(n));
                    counts[compositionKey(blocks)] += 1;
                }
                for (const auto& [key, lw] : exact) {
                    const double p = std::exp(lw - lad.logAlphaN());
                    const double f = static_cast<double>(counts[key]) / nsamp;
                    const double sigma = std::sqrt(p * (1 - p) / nsamp) + 1e-12;
                    check(std::abs(f - p) < 5 * sigma + 1e-4,
                          "sampler frequency, N = " + std::to_string(n) + " composition " + key + ": " + std::to_string(f) + " vs " + std::to_string(p));
                }
            }
        }
    }

    // 4a: all kernels equal: cycle-type frequencies = |C_lambda| / N!  (N = 4 and 6)
    for (int n : {4, 6}) {
        Kernels k;
        k.S.assign(n, 0.0); k.B.assign(n - 1, 0.0); k.R.assign(n - 1, 0.0); k.L.assign(n - 2, 0.0);
        LadderRecursion lad(n);
        lad.compute(k.S, k.B, k.R, k.L);
        check(std::abs(lad.logAlphaN()) < 1e-12, "alpha_N = 1 for unit kernels (sum of class fractions), N = " + std::to_string(n));
        // exact class fractions by enumerating S_n
        std::vector<int> p(n);
        for (int i = 0; i < n; ++i) p[i] = i;
        std::map<std::string, int> cls;
        int total = 0;
        do { cls[cycleTypeKey(p)] += 1; ++total; } while (std::next_permutation(p.begin(), p.end()));
        const int nsamp = 400000;
        std::map<std::string, int> counts;
        std::vector<int> perm(n);
        for (int s = 0; s < nsamp; ++s) { lad.sample(perm, gen); counts[cycleTypeKey(perm)] += 1; }
        for (const auto& [key, c] : cls) {
            const double pe = static_cast<double>(c) / total;
            const double f = static_cast<double>(counts[key]) / nsamp;
            const double sigma = std::sqrt(pe * (1 - pe) / nsamp);
            check(std::abs(f - pe) < 5 * sigma + 1e-4,
                  "class fraction N = " + std::to_string(n) + " type " + key + ": " + std::to_string(f) + " vs " + std::to_string(pe));
        }
    }

    // N = 64 with strongly varying kernels: finite, consistent passes (numerical stability)
    {
        const int n = 64;
        Kernels k = randomKernels(n, gen, 40.0);
        LadderRecursion lad(n);
        lad.compute(k.S, k.B, k.R, k.L);
        check(std::isfinite(lad.logAlphaN()), "N = 64 finite");
        check(std::abs(lad.logAlphaTilde0() - lad.logAlphaN()) < 1e-8 * std::abs(lad.logAlphaN()), "N = 64 passes agree");
        for (int l = 0; l < n; ++l) {
            double out = 0.0;
            for (int u = 0; u < n; ++u) out += lad.connectionProbability(l, u);
            check(std::abs(out - 1.0) < 1e-9, "N = 64 outgoing sum l = " + std::to_string(l));
        }
    }

    if (failures == 0) std::printf("ladder_recursion_test: all checks passed\n");
    else std::printf("ladder_recursion_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
