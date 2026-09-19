#!/usr/bin/env python
"""In this check, we verify the accuracy of the orbital-gradient by comparing it
to forward energy differences.

CI vectors remain fixed. We scan down to 1e-6 and fit one order using all
points except the first four. The fitted order must exceed 0.8.
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, fixed_ci_energy, make_direction,
    ORBITAL_STEPS, orbital_parser, rotate_mos, run_checks,
)


def evaluate(klas, mo_coeff, config, steps, rotation_blocks=None):
    ci = copy_ci(klas.ci)
    blocks = config["rotation_blocks"] if rotation_blocks is None else rotation_blocks
    kappa = make_direction(klas, blocks, config["seed"], mo_coeff=mo_coeff)
    gradient = klas.get_grad_orb(mo_coeff=mo_coeff, ci=ci)
    analytic = np.real(np.vdot(gradient, kappa)) / klas.nkpts
    energy_zero = fixed_ci_energy(klas, mo_coeff, ci)
    differences = [
        (fixed_ci_energy(klas, rotate_mos(mo_coeff, kappa, step), ci)
         - energy_zero) / step
        for step in steps
    ]
    return convergence_result(analytic, differences, steps)


if __name__ == "__main__":
    description = "Orbital gradient: forward energy differences"
    run_checks(
        evaluate, description, parser=orbital_parser(description),
        default_steps=ORBITAL_STEPS,
    )
