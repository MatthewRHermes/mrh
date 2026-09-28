#!/usr/bin/env python
"""In this check, we verify the accuracy of the orbital-gradient by comparing it
to centered energy differences. CI vectors remain fixed. Halving the step size 
should reduce the relative error by a factor of four.
We scramble the orbitals away from the nearly stationary reference. The fitted
step range stops at 6.25e-4, before cancellation in the centered energy
difference dominates.
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, fixed_ci_energy, make_direction,
    orbital_parser, rotate_mos, run_checks,
)

SCRAMBLE_AMPLITUDE = 0.1
# ``assert_convergence`` excludes the first four scan points. Start high enough
# that the retained window spans 1e-2 through 6.25e-4.
CENTERED_ORBITAL_STEPS = 0.16 / 2.0 ** np.arange(9)


def evaluate(klas, mo_coeff, config, steps, rotation_blocks=None):
    ci = copy_ci(klas.ci)
    blocks = config["rotation_blocks"] if rotation_blocks is None else rotation_blocks
    kappa = make_direction(klas, blocks, config["seed"], mo_coeff=mo_coeff)
    mo_coeff = rotate_mos(mo_coeff, kappa, SCRAMBLE_AMPLITUDE)
    gradient = klas.get_grad_orb(mo_coeff=mo_coeff, ci=ci)
    analytic = np.real(np.vdot(gradient, kappa)) / klas.nkpts
    differences = [
        (fixed_ci_energy(klas, rotate_mos(mo_coeff, kappa, step), ci)
         - fixed_ci_energy(klas, rotate_mos(mo_coeff, kappa, -step), ci))
        / (2.0 * step)
        for step in steps
    ]
    return convergence_result(analytic, differences, steps)


if __name__ == "__main__":
    description = "Orbital gradient: centered energy differences"
    run_checks(
        evaluate, description, expected_order=2, parser=orbital_parser(description),
        default_steps=CENTERED_ORBITAL_STEPS, plot_xlim=(1.0, 1e-4),
    )
