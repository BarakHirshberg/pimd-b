#include "moves/exchange_move.h"
#include "simulation.h"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "winding.h"

ExchangeMove::ExchangeMove(Simulation& _sim, int _max_segment, int attempts, int _kmax, unsigned int seed) :
    sim(_sim),
    max_segment(std::min(_max_segment, _sim.nbeads - 2)),
    n_attempts(attempts),
    kmax(std::max(2, std::min(_kmax, _sim.natoms))),
    gen(seed),
    n_trials(0),
    n_accepted(0),
    paths(kmax, std::vector<Vec>(_sim.nbeads)),
    gather_buffer(kmax * NDIM * _sim.nbeads),
    proposal_buffer(kmax * NDIM * std::max(1, std::min(_max_segment, _sim.nbeads - 2))) {
    if (max_segment < 1) {
        throw std::invalid_argument("The exchange move requires at least 3 beads (nbeads >= 3).");
    }
}

/**
 * @brief Minimum-image separation x2 - x1 (consistent with the spring energies of the code).
 */
void ExchangeMove::separation(const Vec& x1, const Vec& x2, Vec& diff) const {
    for (int axis = 0; axis < NDIM; ++axis) {
        double dx = x2[axis] - x1[axis];
#if MINIM
        if (sim.pbc) applyMinimumImage(dx, sim.size);
#endif
        diff[axis] = dx;
    }
}

/**
 * @brief Index (within the block) of the particle onto which particle i closes under closure tau:
 * 0 = identity, 1 = forward cycle (i -> i+1), 2 = backward cycle (i -> i-1).
 */
int ExchangeMove::closureTarget(int i, int k, int tau) {
    if (tau == 0) return i;
    if (tau == 1) return (i + 1) % k;
    return (i - 1 + k) % k;
}

/**
 * @brief Collects the coordinates of the block particles on all beads (identical result on every rank).
 */
void ExchangeMove::gatherPaths(const std::vector<int>& block) {
    const int k = static_cast<int>(block.size());
    std::vector<double> local(k * NDIM);
    for (int i = 0; i < k; ++i) {
        for (int axis = 0; axis < NDIM; ++axis) {
            local[i * NDIM + axis] = sim.coord(block[i], axis);
        }
    }
    MPI_Allgather(local.data(), k * NDIM, MPI_DOUBLE, gather_buffer.data(), k * NDIM, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int j = 0; j < sim.nbeads; ++j) {
        for (int i = 0; i < k; ++i) {
            for (int axis = 0; axis < NDIM; ++axis) {
                paths[i][j][axis] = gather_buffer[k * NDIM * j + i * NDIM + axis];
            }
        }
    }
}

/**
 * @brief Unwrapped target of a bridge from anchor towards end: anchor + MIC(end - anchor), plus (with
 * winding-sum springs, when draw_image is set) an image shift w L per component drawn from the periodic
 * free propagator over the m+1 links, which makes the proposal density of the regrown segment the
 * image-summed one.
 */
ExchangeMove::Vec ExchangeMove::bridgeTarget(const Vec& anchor, const Vec& end, int m, bool draw_image) {
    Vec diff{}, target{};
    separation(anchor, end, diff);
    for (int axis = 0; axis < NDIM; ++axis) {
        double shift = 0.0;
        if (sim.winding_springs && draw_image) {
            const WindingProbability wp(diff[axis], sim.max_wind, sim.beta_half_k / (m + 1), sim.size);
            shift = sim.size * wp.sample(gen);
        }
        target[axis] = anchor[axis] + diff[axis] + shift;
    }
    return target;
}

/**
 * @brief Free-particle (Levy) bridge of m beads between the fixed points start and target (m+1 links).
 * The single-link variance is 1/(beta_P k) with k the ring-polymer spring constant.
 */
void ExchangeMove::levyBridge(const Vec& start, const Vec& target, int m, std::vector<Vec>& out) {
    const double sigma2 = 1.0 / (sim.thermo_beta * sim.spring_constant);
    std::normal_distribution<double> normal(0.0, 1.0);

    out.assign(m, {});
    Vec prev = start;
    for (int j = 1; j <= m; ++j) {
        const int remaining = m + 2 - j;  // links left including the one being drawn
        const double var = sigma2 * (remaining - 1) / static_cast<double>(remaining);
        for (int axis = 0; axis < NDIM; ++axis) {
            const double mean = prev[axis] + (target[axis] - prev[axis]) / remaining;
            out[j - 1][axis] = mean + std::sqrt(var) * normal(gen);
        }
        prev = out[j - 1];
    }
}

/**
 * @brief Energy of a closing link given by the RAW (unwrapped) difference target - last: (k/2) d^2 with
 * minimum-image springs, -(1/beta_P) ln mu(d) with winding-sum springs (mu is periodic, so the image of
 * the target is summed over consistently with the forward proposal).
 */
double ExchangeMove::closingEnergy(const Vec& last, const Vec& target) const {
    double raw[NDIM];
    for (int axis = 0; axis < NDIM; ++axis) raw[axis] = target[axis] - last[axis];
    if (sim.winding_springs) return -sim.linkLogWeight(raw) / sim.thermo_beta;
    double s = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) s += raw[axis] * raw[axis];
    return 0.5 * sim.spring_constant * s;
}

/**
 * @brief "Energy" of the free propagator of the anchor -> end minimum-image displacement over m+1 links:
 * k D^2 / (2 (m+1)), or its image-summed counterpart with winding-sum springs.
 */
double ExchangeMove::propagatorEnergy(const Vec& anchor, const Vec& end, int m) const {
    Vec d{};
    separation(anchor, end, d);
    if (sim.winding_springs) {
        double lw = 0.0;
        for (int axis = 0; axis < NDIM; ++axis) {
            lw += WindingProbability(d[axis], sim.max_wind, sim.beta_half_k / (m + 1), sim.size).logWeight();
        }
        return -lw / sim.thermo_beta;
    }
    double s = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) s += d[axis] * d[axis];
    return 0.5 * sim.spring_constant * s / (m + 1);
}

/**
 * @brief Change of the physical potential on this rank's slice when the block particles move to new_pos.
 */
double ExchangeMove::slicePotentialChange(const std::vector<int>& block, const std::vector<Vec>& new_pos) const {
    const int k = static_cast<int>(block.size());
    std::vector<int> in_block(sim.natoms, -1);
    for (int i = 0; i < k; ++i) in_block[block[i]] = i;

    auto energy = [&](const std::vector<Vec>& pos) {
        double e = 0.0;
        if (sim.int_pot_cutoff != 0.0) {
            auto pair = [&](const Vec& x, const Vec& y) {
                dVec d;
                for (int axis = 0; axis < NDIM; ++axis) {
                    double dx = x[axis] - y[axis];
                    if (sim.pbc && MINIM) applyMinimumImage(dx, sim.size);
                    d(0, axis) = dx;
                }
                const double r = d.norm();
                return (r < sim.int_pot_cutoff || sim.int_pot_cutoff < 0.0) ? sim.int_potential->V(d) : 0.0;
            };
            // Block particles with the rest of the system
            for (int i = 0; i < k; ++i) {
                for (int p = 0; p < sim.natoms; ++p) {
                    if (in_block[p] >= 0) continue;
                    Vec y{};
                    for (int axis = 0; axis < NDIM; ++axis) y[axis] = sim.coord(p, axis);
                    e += pair(pos[i], y);
                }
            }
            // Within the block
            for (int i = 0; i < k; ++i) {
                for (int j = i + 1; j < k; ++j) {
                    e += pair(pos[i], pos[j]);
                }
            }
        }
        if (sim.external_potential_name != "free") {
            dVec xs(k);
            for (int i = 0; i < k; ++i) {
                for (int axis = 0; axis < NDIM; ++axis) xs(i, axis) = pos[i][axis];
            }
            e += sim.ext_potential->V(xs);
        }
        return e;
    };

    std::vector<Vec> old_pos(k);
    for (int i = 0; i < k; ++i) {
        for (int axis = 0; axis < NDIM; ++axis) old_pos[i][axis] = sim.coord(block[i], axis);
    }
    return energy(new_pos) - energy(old_pos);
}

void ExchangeMove::attempt() {
    const int P = sim.nbeads;
    const int N = sim.natoms;

    for (int t = 0; t < n_attempts; ++t) {
        // ---- proposal drawn on rank 0: block start a, block size k, segment length m, closure tau ----
        int header[4] = {0, 2, 1, 0};
        if (sim.this_bead == 0) {
            std::uniform_int_distribution<int> pick_k(2, kmax);
            const int k = pick_k(gen);
            std::uniform_int_distribution<int> pick_a(0, N - k);
            std::uniform_int_distribution<int> pick_m(1, max_segment);
            std::uniform_int_distribution<int> pick_tau(0, k == 2 ? 1 : 2);
            header[0] = pick_a(gen);
            header[1] = k;
            header[2] = pick_m(gen);
            header[3] = pick_tau(gen);
        }
        MPI_Bcast(header, 4, MPI_INT, 0, MPI_COMM_WORLD);
        const int a = header[0], k = header[1], m = header[2], tau = header[3];
        const int n_closures = (k == 2) ? 2 : 3;

        std::vector<int> block(k);
        for (int i = 0; i < k; ++i) block[i] = a + i;  // consecutive labels: where the recursion has weight

        gatherPaths(block);

        // Anchors (bead P-m-1) and first beads (bead 0) of the block
        std::vector<Vec> anchor(k), first(k), old_last(k);
        for (int i = 0; i < k; ++i) {
            anchor[i] = paths[i][P - m - 1];
            first[i] = paths[i][0];
            old_last[i] = paths[i][P - 1];
        }

        // Bridges on rank 0 (targets for the chosen closure; image drawn with winding-sum springs)
        std::vector<std::vector<Vec>> new_seg(k);
        if (sim.this_bead == 0) {
            for (int i = 0; i < k; ++i) {
                const Vec target = bridgeTarget(anchor[i], first[closureTarget(i, k, tau)], m, true);
                levyBridge(anchor[i], target, m, new_seg[i]);
                for (int j = 0; j < m; ++j) {
                    for (int axis = 0; axis < NDIM; ++axis) {
                        proposal_buffer[(j * k + i) * NDIM + axis] = new_seg[i][j][axis];
                    }
                }
            }
        }
        MPI_Bcast(proposal_buffer.data(), k * NDIM * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (sim.this_bead != 0) {
            for (int i = 0; i < k; ++i) {
                new_seg[i].assign(m, {});
                for (int j = 0; j < m; ++j) {
                    for (int axis = 0; axis < NDIM; ++axis) {
                        new_seg[i][j][axis] = proposal_buffer[(j * k + i) * NDIM + axis];
                    }
                }
            }
        }

        // ---- physical potential change on the regrown slices (ranks P-m .. P-1) ----
        double local_du = 0.0;
        const int seg_index = sim.this_bead - (P - m);  // position of this rank's bead within the segment
        std::vector<Vec> new_here(k);
        if (seg_index >= 0 && seg_index < m) {
            for (int i = 0; i < k; ++i) new_here[i] = new_seg[i][seg_index];
            local_du = slicePotentialChange(block, new_here);
        }
        double du = 0.0;
        MPI_Reduce(&local_du, &du, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // ---- exterior potential change and closing-link / propagator terms on rank 0 ----
        int accept = 0;
        if (sim.this_bead == 0) {
            const double v_old = sim.bosonic_exchange->effectivePotential();

            std::vector<Vec> saved(k);
            for (int i = 0; i < k; ++i) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    saved[i][axis] = sim.prev_coord(block[i], axis);
                    sim.prev_coord(block[i], axis) = new_seg[i][m - 1][axis];
                }
            }
            sim.bosonic_exchange->prepare();
            const double v_new = sim.bosonic_exchange->effectivePotential();

            // Mixture proposal over the closures: S(seg) = sum_tau' exp(-beta_P [E_close,tau'(seg) - g_tau']),
            // where E_close uses the RAW closing links to the (image-free) bridge targets of each closure and
            // g_tau' the free-propagator energies of the anchor-to-end displacements. The interior spring
            // energies of the segment cancel between target and proposal.
            std::vector<double> e_old(n_closures), e_new(n_closures);
            for (int c = 0; c < n_closures; ++c) {
                double g = 0.0, eo = 0.0, en = 0.0;
                for (int i = 0; i < k; ++i) {
                    const Vec& end = first[closureTarget(i, k, c)];
                    const Vec target = bridgeTarget(anchor[i], end, m, false);
                    g += propagatorEnergy(anchor[i], end, m);
                    eo += closingEnergy(old_last[i], target);
                    en += closingEnergy(new_seg[i][m - 1], target);
                }
                e_old[c] = sim.thermo_beta * (eo - g);
                e_new[c] = sim.thermo_beta * (en - g);
            }
            auto log_sum_exp = [](const std::vector<double>& x) {
                const double mn = *std::min_element(x.begin(), x.end());
                double s = 0.0;
                for (double v : x) s += std::exp(-(v - mn));
                return -mn + std::log(s);
            };
            const double log_accept = -sim.thermo_beta * (du + (v_new - v_old)) + (log_sum_exp(e_old) - log_sum_exp(e_new));
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            accept = (log_accept >= 0.0 || uniform(gen) < std::exp(log_accept)) ? 1 : 0;

            if (!accept) {
                for (int i = 0; i < k; ++i) {
                    for (int axis = 0; axis < NDIM; ++axis) sim.prev_coord(block[i], axis) = saved[i][axis];
                }
                sim.bosonic_exchange->prepare();
            }
            n_trials += 1;
            n_accepted += accept;
        }
        MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (accept) {
            if (seg_index >= 0 && seg_index < m) {
                for (int i = 0; i < k; ++i) {
                    for (int axis = 0; axis < NDIM; ++axis) sim.coord(block[i], axis) = new_here[i][axis];
                }
            }
            sim.updateNeighboringCoordinates();
            sim.updateForces();
            sim.configurationChanged();
        }
    }
}

double ExchangeMove::acceptanceRatio() const {
    return (n_trials > 0) ? static_cast<double>(n_accepted) / n_trials : 0.0;
}
