#!/usr/bin/env python
"""In this check, we verify the accuracy of the orbital Hessian-vector product
by comparing it to forward orbital-gradient differences.

CI vectors remain fixed. We account for the moving orbital frame and the
packing factor. The relative error should decrease linearly with the step size.
We scan down to 1e-6 and fit all points except the first four; the order must exceed 0.8.
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, make_direction, orbital_parser, rotate_mos, run_checks,
    ORBITAL_STEPS,
)


def evaluate(klas, mo_coeff, config, steps, rotation_blocks=None):
    ci = copy_ci(klas.ci)
    ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    hop = klas.get_hop(mo_coeff=mo_coeff, ci=ci, ugg=ugg)
    hop.level_shift = 0.0
    blocks = config["rotation_blocks"] if rotation_blocks is None else rotation_blocks
    kappa = make_direction(klas, blocks, config["seed"], mo_coeff=mo_coeff)
    trial = np.zeros(ugg.nvar_tot, dtype=complex)
    trial[:ugg.nvar_orb] = ugg.pack_orb(kappa)
    analytic = 2.0 * hop.matvec(trial)[:ugg.nvar_orb]

    connection = np.asarray([
        (fock @ generator - generator @ fock) / 2.0
        for fock, generator in zip(hop.fock1, kappa)
    ])
    connection -= connection.conj().transpose(0, 2, 1)
    connection = ugg.pack_orb(connection)
    gradient_zero = ugg.pack_orb(klas.get_grad_orb(mo_coeff=mo_coeff, ci=ci))
    differences = []
    for step in steps:
        gradient = ugg.pack_orb(klas.get_grad_orb(
            mo_coeff=rotate_mos(mo_coeff, kappa, step), ci=ci,
        ))
        differences.append((gradient - gradient_zero) / step - connection)
    return convergence_result(analytic, differences, steps)


if __name__ == "__main__":
    description = "Orbital Hessian-vector product: forward gradient differences"
    run_checks(
        evaluate, description, parser=orbital_parser(description),
        default_steps=ORBITAL_STEPS,
    )
