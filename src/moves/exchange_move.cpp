#include "moves/exchange_move.h"
#include "simulation.h"
#include "mpi.h"

#include <algorithm>
#include <cmath>

ExchangeMove::ExchangeMove(Simulation& _sim, int _max_segment, int attempts, unsigned int seed) :
    sim(_sim),
    max_segment(std::min(_max_segment, _sim.nbeads - 2)),
    n_attempts(attempts),
    gen(seed),
    n_trials(0),
    n_accepted(0),
    path_a(_sim.nbeads),
    path_b(_sim.nbeads),
    gather_buffer(2 * NDIM * _sim.nbeads),
    proposal_buffer(4 + 2 * NDIM * std::max(1, std::min(_max_segment, _sim.nbeads - 2))) {
    if (max_segment < 1) {
        throw std::invalid_argument("The exchange move requires at least 3 beads (nbeads >= 3).");
    }
}

/**
 * @brief Minimum-image separation x2 - x1 (consistent with the spring energies of the code).
 */
void ExchangeMove::separation(const std::array<double, NDIM>& x1, const std::array<double, NDIM>& x2,
                              std::array<double, NDIM>& diff) const {
    for (int axis = 0; axis < NDIM; ++axis) {
        double dx = x2[axis] - x1[axis];
#if MINIM
        if (sim.pbc) applyMinimumImage(dx, sim.size);
#endif
        diff[axis] = dx;
    }
}

/**
 * @brief Collects the coordinates of particles a and b on all beads (identical result on every rank).
 */
void ExchangeMove::gatherPaths(int a, int b) {
    std::array<double, 2 * NDIM> local{};
    for (int axis = 0; axis < NDIM; ++axis) {
        local[axis] = sim.coord(a, axis);
        local[NDIM + axis] = sim.coord(b, axis);
    }
    MPI_Allgather(local.data(), 2 * NDIM, MPI_DOUBLE, gather_buffer.data(), 2 * NDIM, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int j = 0; j < sim.nbeads; ++j) {
        for (int axis = 0; axis < NDIM; ++axis) {
            path_a[j][axis] = gather_buffer[2 * NDIM * j + axis];
            path_b[j][axis] = gather_buffer[2 * NDIM * j + NDIM + axis];
        }
    }
}

/**
 * @brief Free-particle (Levy) bridge of m beads between fixed points start and end (m+1 links).
 * The single-link variance is 1/(beta_P k) with k the ring-polymer spring constant. The bridge is
 * built in unwrapped coordinates from the minimum-image displacement between start and end.
 */
void ExchangeMove::levyBridge(const std::array<double, NDIM>& start, const std::array<double, NDIM>& end, int m,
                              std::vector<std::array<double, NDIM>>& out) {
    const double sigma2 = 1.0 / (sim.thermo_beta * sim.spring_constant);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::array<double, NDIM> target{};
    std::array<double, NDIM> diff{};
    separation(start, end, diff);
    for (int axis = 0; axis < NDIM; ++axis) target[axis] = start[axis] + diff[axis];

    out.assign(m, {});
    std::array<double, NDIM> prev = start;
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
 * @brief Change of the physical potential on this rank's slice when particles a and b move to new_a, new_b.
 */
double ExchangeMove::slicePotentialChange(int a, int b, const std::array<double, NDIM>& new_a,
                                          const std::array<double, NDIM>& new_b) const {
    // Pair interactions with all other particles (and between a and b), plus the external potential
    auto pair_energy = [&](const std::array<double, NDIM>& xa, const std::array<double, NDIM>& xb) {
        double e = 0.0;
        if (sim.int_pot_cutoff != 0.0) {
            for (int p = 0; p < sim.natoms; ++p) {
                if (p == a || p == b) continue;
                for (int which = 0; which < 2; ++which) {
                    const auto& x = which == 0 ? xa : xb;
                    dVec d;
                    for (int axis = 0; axis < NDIM; ++axis) {
                        double dx = x[axis] - sim.coord(p, axis);
                        if (sim.pbc && MINIM) applyMinimumImage(dx, sim.size);
                        d(0, axis) = dx;
                    }
                    const double r = d.norm();
                    if (r < sim.int_pot_cutoff || sim.int_pot_cutoff < 0.0) e += sim.int_potential->V(d);
                }
            }
            dVec dab;
            for (int axis = 0; axis < NDIM; ++axis) {
                double dx = xa[axis] - xb[axis];
                if (sim.pbc && MINIM) applyMinimumImage(dx, sim.size);
                dab(0, axis) = dx;
            }
            const double r = dab.norm();
            if (r < sim.int_pot_cutoff || sim.int_pot_cutoff < 0.0) e += sim.int_potential->V(dab);
        }
        if (sim.external_potential_name != "free") {
            dVec two(2);
            for (int axis = 0; axis < NDIM; ++axis) {
                two(0, axis) = xa[axis];
                two(1, axis) = xb[axis];
            }
            e += sim.ext_potential->V(two);
        }
        return e;
    };

    std::array<double, NDIM> old_a{}, old_b{};
    for (int axis = 0; axis < NDIM; ++axis) {
        old_a[axis] = sim.coord(a, axis);
        old_b[axis] = sim.coord(b, axis);
    }
    return pair_energy(new_a, new_b) - pair_energy(old_a, old_b);
}

void ExchangeMove::attempt() {
    const int P = sim.nbeads;
    const int N = sim.natoms;
    const double k_half = 0.5 * sim.spring_constant;

    for (int t = 0; t < n_attempts; ++t) {
        // ---- proposal drawn on rank 0 ----
        int a = 0, b = 1, m = 1, swap = 0;
        if (sim.this_bead == 0) {
            std::uniform_int_distribution<int> pick_pair(0, N - 2);
            std::uniform_int_distribution<int> pick_m(1, max_segment);
            std::uniform_int_distribution<int> pick_target(0, 1);
            a = pick_pair(gen);
            b = a + 1;  // consecutive labels: where the quadratic recursion assigns exchange weight
            m = pick_m(gen);
            swap = pick_target(gen);
        }
        int header[4] = {a, b, m, swap};
        MPI_Bcast(header, 4, MPI_INT, 0, MPI_COMM_WORLD);
        a = header[0]; b = header[1]; m = header[2]; swap = header[3];

        gatherPaths(a, b);

        // Anchors (bead P-m, index P-m-1) and first beads (index 0)
        const std::array<double, NDIM>& anchor_a = path_a[P - m - 1];
        const std::array<double, NDIM>& anchor_b = path_b[P - m - 1];
        const std::array<double, NDIM>& first_a = path_a[0];
        const std::array<double, NDIM>& first_b = path_b[0];
        const std::array<double, NDIM>& end_a = swap ? first_b : first_a;
        const std::array<double, NDIM>& end_b = swap ? first_a : first_b;

        std::vector<std::array<double, NDIM>> new_a, new_b;
        if (sim.this_bead == 0) {
            levyBridge(anchor_a, end_a, m, new_a);
            levyBridge(anchor_b, end_b, m, new_b);
            for (int j = 0; j < m; ++j) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    proposal_buffer[2 * NDIM * j + axis] = new_a[j][axis];
                    proposal_buffer[2 * NDIM * j + NDIM + axis] = new_b[j][axis];
                }
            }
        }
        MPI_Bcast(proposal_buffer.data(), 2 * NDIM * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (sim.this_bead != 0) {
            new_a.assign(m, {});
            new_b.assign(m, {});
            for (int j = 0; j < m; ++j) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    new_a[j][axis] = proposal_buffer[2 * NDIM * j + axis];
                    new_b[j][axis] = proposal_buffer[2 * NDIM * j + NDIM + axis];
                }
            }
        }

        // ---- physical potential change on the regrown slices (ranks P-m .. P-1) ----
        double local_du = 0.0;
        const int seg_index = sim.this_bead - (P - m);  // position of this rank's bead within the segment
        if (seg_index >= 0 && seg_index < m) {
            local_du = slicePotentialChange(a, b, new_a[seg_index], new_b[seg_index]);
        }
        double du = 0.0;
        MPI_Reduce(&local_du, &du, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // ---- exterior potential change and closing-link / propagator terms on rank 0 ----
        int accept = 0;
        if (sim.this_bead == 0) {
            const double v_old = sim.bosonic_exchange->effectivePotential();

            std::array<double, NDIM> saved_a{}, saved_b{};
            for (int axis = 0; axis < NDIM; ++axis) {
                saved_a[axis] = sim.prev_coord(a, axis);
                saved_b[axis] = sim.prev_coord(b, axis);
                sim.prev_coord(a, axis) = new_a[m - 1][axis];
                sim.prev_coord(b, axis) = new_b[m - 1][axis];
            }
            sim.bosonic_exchange->prepare();
            const double v_new = sim.bosonic_exchange->effectivePotential();

            // Mixture proposal: the target (identity or exchange) is drawn with probability 1/2, so the
            // proposal density of a segment is q(seg) = (1/2) sum_tau B_tau(seg) with
            // B_tau(seg) = exp(-beta_P [S_int(seg) + E_close,tau(seg)]) / G_tau, G_tau the free propagator of
            // the anchor-to-target displacement over m+1 links. In the Metropolis ratio the interior spring
            // energies S_int cancel between pi and q, leaving
            //   A = exp(-beta_P (dU + dV_B)) * sum_tau exp(-beta_P (E_close,tau(old) - g_tau))
            //                                / sum_tau exp(-beta_P (E_close,tau(new) - g_tau)),
            // with g_tau = k/(2(m+1)) |end_tau - anchor|^2 summed over both particles (log-sum-exp below).
            std::array<double, NDIM> d{};
            auto sq = [&](const std::array<double, NDIM>& x1, const std::array<double, NDIM>& x2) {
                separation(x1, x2, d);
                double s = 0.0;
                for (int axis = 0; axis < NDIM; ++axis) s += d[axis] * d[axis];
                return s;
            };
            const double k_bridge = k_half / (m + 1);
            const double g_id = k_bridge * (sq(anchor_a, first_a) + sq(anchor_b, first_b));
            const double g_sw = k_bridge * (sq(anchor_a, first_b) + sq(anchor_b, first_a));
            const double old_id = k_half * (sq(path_a[P - 1], first_a) + sq(path_b[P - 1], first_b)) - g_id;
            const double old_sw = k_half * (sq(path_a[P - 1], first_b) + sq(path_b[P - 1], first_a)) - g_sw;
            const double new_id = k_half * (sq(new_a[m - 1], first_a) + sq(new_b[m - 1], first_b)) - g_id;
            const double new_sw = k_half * (sq(new_a[m - 1], first_b) + sq(new_b[m - 1], first_a)) - g_sw;
            auto log_sum_exp = [&](double x1, double x2) {
                const double mn = std::min(x1, x2);
                return -mn + std::log(std::exp(-(x1 - mn)) + std::exp(-(x2 - mn)));
            };
            // ln sum_tau exp(-beta_P e_tau) for old and new segments
            const double ls_old = log_sum_exp(sim.thermo_beta * old_id, sim.thermo_beta * old_sw);
            const double ls_new = log_sum_exp(sim.thermo_beta * new_id, sim.thermo_beta * new_sw);

            const double log_accept = -sim.thermo_beta * (du + (v_new - v_old)) + (ls_old - ls_new);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            accept = (log_accept >= 0.0 || uniform(gen) < std::exp(log_accept)) ? 1 : 0;

            if (!accept) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    sim.prev_coord(a, axis) = saved_a[axis];
                    sim.prev_coord(b, axis) = saved_b[axis];
                }
                sim.bosonic_exchange->prepare();
            }
            n_trials += 1;
            n_accepted += accept;
        }
        MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (accept) {
            if (seg_index >= 0 && seg_index < m) {
                for (int axis = 0; axis < NDIM; ++axis) {
                    sim.coord(a, axis) = new_a[seg_index][axis];
                    sim.coord(b, axis) = new_b[seg_index][axis];
                }
            }
            sim.updateNeighboringCoordinates();
            sim.updateForces();
        }
    }
}

double ExchangeMove::acceptanceRatio() const {
    return (n_trials > 0) ? static_cast<double>(n_accepted) / n_trials : 0.0;
}
