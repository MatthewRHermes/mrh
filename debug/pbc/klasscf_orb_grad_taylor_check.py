#!/usr/bin/env python
r"""In this script, we check \mathcal{O}(\delta) relative orbital-gradient error 
from forward energy differences.

With fixed CI and C(delta) = C exp(delta*kappa), the per-cell derivative is
Re(<g_orb, kappa>)/nkpts. Equivalently, the energy Taylor residual is \mathcal{O}(\delta**2).

The default step-size scan halves delta from 0.5, ending at 1e-6:
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, fixed_ci_energy, make_direction,
    orbital_parser, rotate_mos, run_checks,
)

SCAN_STEPS = np.append(0.5 ** np.arange(1, 20), 1e-6)


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
        default_steps=SCAN_STEPS,
    )
