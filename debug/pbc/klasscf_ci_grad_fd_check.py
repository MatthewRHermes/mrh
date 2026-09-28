#!/usr/bin/env python
"""In this check, we verify the accuracy of the CI-gradient by comparing it to
forward energy differences.Orbitals remain translation symmetric and cell CI vectors may
vary independently.
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, displace_ci, fixed_ci_energy,
    make_ci_direction, run_checks,
)

def evaluate(klas, mo_coeff, config, steps):
    ci = copy_ci(klas.ci)
    ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    _, displacement = make_ci_direction(ugg, config["seed"] + 100)
    ci = displace_ci(ci, displacement, 0.1)
    ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    packed, direction = make_ci_direction(ugg, config["seed"])
    h2eff = klas.get_h2cas(mo_coeff)
    gradient = ugg.pack_ci(klas.get_grad_ci(
        mo_coeff=mo_coeff, ci=ci, h2eff=h2eff,
    ))
    analytic = np.real(np.vdot(gradient, packed)) / klas.nkpts

    energy_zero = fixed_ci_energy(klas, mo_coeff, ci)

    differences = [
        (fixed_ci_energy(klas, mo_coeff, displace_ci(ci, direction, step))
         - energy_zero) / step
        for step in steps
    ]
    return convergence_result(analytic, differences, steps)


if __name__ == "__main__":
    run_checks(evaluate, "CI gradient: forward energy differences")
