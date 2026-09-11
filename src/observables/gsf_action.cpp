#include "observables/gsf_action.h"
#include "simulation.h"
#include "units.h"
#include <ranges>
#include "mpi.h"

/**
 * @brief Constructor for the class handling observables associated with the GSF action.
 *
 * @param _extra If true, the observable also reports even_pot_gsf and kin_gsf (see calculate()).
 */
GSFActionObservable::GSFActionObservable(const Simulation& _sim, int _freq, const std::string& _out_unit, bool _extra) :
    Observable(_sim, _freq, _out_unit), extra(_extra) {
    if (extra) {
        initialize({ "w_gsf", "pot_gsf", "even_pot_gsf", "kin_gsf" });
    } else {
        initialize({ "w_gsf", "pot_gsf" });
    }
}

/**
 * @brief Calculates the natural logarithm of the weight associated with the GSF action,
 * which is used for re-weighting the observables [See J. Chem. Phys. 135, 064104 (2011) and
 * Higer & Hirshberg, Phys. Chem. Chem. Phys. 28, 17846 (2026)]:
 *
 *   ln w_GSF = -beta * sum_s [ (V(2s) - V(2s-1)) / (3P)
 *                             + (1/(9 m omega_P^2 P^2)) sum_l ( alpha F_l^2(2s-1) + (1-alpha) F_l^2(2s) ) ],
 *
 * where the sum over pairs of slices is realised here as a sum over beads with odd/even parity (the
 * per-bead contributions are added across ranks by the logger). Also calculates the potential energy
 * estimator restricted to odd imaginary-time slices (pot_gsf) and, if requested, the even-slice
 * potential (even_pot_gsf) and the force-squared correction to the primitive kinetic energy estimator
 * (kin_gsf, PCCP Eq. 18): <K>_GSF = dNP/(2 beta) - <springs> + <kin_gsf>, all re-weighted with w_GSF.
 * The tuning parameter alpha (0 <= alpha <= 1) is read from [observables] gsf_alpha (default 0).
 */
void GSFActionObservable::calculate() {
    const double alpha = sim.gsf_alpha;

    double total_potential = sim.ext_potential->V(sim.coord);

    dVec gradients(sim.natoms);
    gradients = sim.ext_potential->gradV(sim.coord);

    if (sim.int_pot_cutoff != 0.0) {
        for (int ptcl_one = 0; ptcl_one < sim.natoms; ++ptcl_one) {
            for (int ptcl_two = ptcl_one + 1; ptcl_two < sim.natoms; ++ptcl_two) {
                dVec diff = sim.getSeparation(ptcl_one, ptcl_two, MINIM);  // Vectorial distance

                if (const double distance = diff.norm(); distance < sim.int_pot_cutoff || sim.int_pot_cutoff < 0.0) {
                    total_potential += sim.int_potential->V(diff);
                    // Pair gradient with respect to the first particle; opposite sign for the second
                    const dVec grad = sim.int_potential->gradV(diff);
                    for (int axis = 0; axis < NDIM; ++axis) {
                        gradients(ptcl_one, axis) += grad(0, axis);
                        gradients(ptcl_two, axis) -= grad(0, axis);
                    }
                }
            }
        }
    }

    double total_force_squared = 0.0;

    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
        for (int axis = 0; axis < NDIM; ++axis) {
            total_force_squared += gradients(ptcl_idx, axis) * gradients(ptcl_idx, axis);
        }
    }

    // Ensure the spring constant is in Tuckerman's convention
#if IPI_CONVENTION
    double sp_constant = sim.spring_constant / sim.nbeads;
#else
    double sp_constant = sim.spring_constant;
#endif

    double potential_term = total_potential / (3 * sim.nbeads);
    double force_squared_term = total_force_squared / (9 * sp_constant * sim.nbeads * sim.nbeads);

    double kinetic_correction;
    if (sim.this_bead % 2 != 0) {
        // Odd
        quantities["w_gsf"] = (-1.0) * potential_term + alpha * force_squared_term;
        kinetic_correction = alpha * force_squared_term;

        // Evaluate the potential energy estimator only for odd imaginary-time slices
        quantities["pot_gsf"] = Units::convertToUser("energy", out_unit, total_potential / (0.5 * sim.nbeads));
    } else {
        // Even
        quantities["w_gsf"] = potential_term + (1 - alpha) * force_squared_term;
        kinetic_correction = (1 - alpha) * force_squared_term;
        if (extra) {
            quantities["even_pot_gsf"] = Units::convertToUser("energy", out_unit, total_potential / (0.5 * sim.nbeads));
        }
    }

    if (extra) {
        quantities["kin_gsf"] = Units::convertToUser("energy", out_unit, kinetic_correction);
    }

    quantities["w_gsf"] *= (-1.0) * sim.beta;
}
