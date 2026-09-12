#pragma once

#include "common.h"
#include "potentials/potential.h"

/* -------------- Aziz potential -------------- */
class AzizPotential : public Potential {
public:
    /**
     * @param v_cap Smooth cap on the repulsive core: V -> v_cap * tanh(V / v_cap), which leaves the
     * potential untouched where |V| << v_cap and saturates it at v_cap at contact. Zero (default)
     * disables the cap and reproduces the original Aziz potential bit for bit. The cap exists to test
     * the sampling theory of docs/09: a bounded core lets the permutation barrier fall as 1/P, while a
     * diverging core makes it grow with P. It changes the physical system, so production runs use 0.
     */
    explicit AzizPotential(double v_cap = 0.0);
    ~AzizPotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double rm, A, epsilon, alpha, D, C6, C8, C10;
    double v_cap;  // 0 = no cap; otherwise V -> v_cap tanh(V / v_cap) (atomic units)

    // Raw (uncapped) Aziz potential and its radial derivative at the scaled distance x = r / rm
    double rawV(double x) const;
    double rawdVdx(double x) const;

    // The auxiliary F-function for the Aziz potential
    double F(const double x) const {
        return (x < D ? exp(-(D / x - 1.0) * (D / x - 1.0)) : 1.0);
    }

    // The derivative of the F-function
    double dF(const double x) const {
        double ix = 1.0 / x;
        double r = 2.0 * D * ix * ix * (D * ix - 1.0) * exp(-(D * ix - 1.0) * (D * ix - 1.0));
        return (x < D ? r : 0.0);
    }

    // The 2nd derivative of the F-function
    double d2F(const double x) const {
        double ix = 1.0 / x;
        double r = 2.0 * D * ix * ix * ix * (2.0 * D * D * D * ix * ix * ix - 4.0 * D * D * ix * ix
            - D * ix + 2.0) * exp(-(D * ix - 1.0) * (D * ix - 1.0));
        return (x < D ? r : 0.0);
    }
};