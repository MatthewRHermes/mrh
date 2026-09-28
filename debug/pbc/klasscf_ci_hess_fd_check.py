#!/usr/bin/env python
"""In this check, we verify the accuracy of the CI Hessian-vector product by
comparing it to forward CI-gradient differences. Orbitals remain fixed and the
CI reference is stationary. The relative error should decrease linearly with the 
step size.
"""

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    convergence_result, copy_ci, displace_ci, make_ci_direction, run_checks,
)


def evaluate(klas, mo_coeff, config, steps):
    ci = copy_ci(klas.ci)
    ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    h2eff = klas.get_h2cas(mo_coeff)
    hop = klas.get_hop(
        mo_coeff=mo_coeff, ci=ci, ugg=ugg, h2eff=h2eff,
    )
    hop.level_shift = 0.0
    packed, direction = make_ci_direction(ugg, config["seed"])
    trial = np.zeros(ugg.nvar_tot, dtype=complex)
    trial[ugg.nvar_orb:] = packed
    analytic = hop.matvec(trial)[ugg.nvar_orb:]
    gradient_zero = ugg.pack_ci(klas.get_grad_ci(
        mo_coeff=mo_coeff, ci=ci, h2eff=h2eff,
    ))
    differences = []
    for step in steps:
        ci_step = displace_ci(ci, direction, step)
        gradient = ugg.pack_ci(klas.get_grad_ci(
            mo_coeff=mo_coeff, ci=ci_step, h2eff=h2eff,
        ))
        differences.append((gradient - gradient_zero) / step)
    return convergence_result(analytic, differences, steps)


if __name__ == "__main__":
    run_checks(evaluate, "CI Hessian-vector product: forward gradient differences")
