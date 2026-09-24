"""Check orbital and CI forward-difference convergence for periodic LiH.

Fit one order over all scan points except the first four and require it above 0.8.
Errors are divided by the numerical energy or gradient change.
Both analytic and numerical norms must be nonzero.
Orbital plots extend to 1e-6.
"""

import unittest
from pathlib import Path

import numpy as np

from mrh.debug.pbc.klasscf_fd_common import (
    CASES, FD_STEPS, ORBITAL_STEPS, assert_convergence, build_reference, plot_convergence,
    copy_ci, displace_ci, make_ci_direction, make_product_solver,
)
from mrh.debug.pbc import (
    klasscf_ci_grad_fd_check,
    klasscf_ci_hess_fd_check,
    klasscf_orb_grad_taylor_check,
    klasscf_orb_hess_fd_check,
)


class DerivativeConvergenceTests(unittest.TestCase):

    plot_dir = Path(".")

    @classmethod
    def setUpClass(cls):
        cls.references = {}

    def check_convergence(self, evaluate, steps=FD_STEPS, **kwargs):
        results = {}
        try:
            for name, config in CASES.items():
                with self.subTest(dimension=name):
                    if name not in self.references:
                        self.references[name] = build_reference(
                            config["lattice"], config["kmesh"],
                        )
                    klas, mo_coeff = self.references[name]
                    result = evaluate(klas, mo_coeff, config, steps, **kwargs)
                    results[name] = result
                    assert_convergence(result)
        finally:
            if self.plot_dir is not None and results:
                description = self._testMethodName.removeprefix("test_").replace("_", " ")
                plot_convergence(
                    results, self.plot_dir / f"{self._testMethodName}.png", description,
                )

    def test_orbital_gradient_forward_difference_order(self):
        self.check_convergence(klasscf_orb_grad_taylor_check.evaluate, steps=ORBITAL_STEPS)

    def test_active_active_gradient_forward_difference_order(self):
        self.check_convergence(
            klasscf_orb_grad_taylor_check.evaluate,
            steps=ORBITAL_STEPS,
            rotation_blocks=("active-active",),
        )

    def test_active_active_hessian_forward_difference_order(self):
        self.check_convergence(
            klasscf_orb_hess_fd_check.evaluate,
            steps=ORBITAL_STEPS,
            rotation_blocks=("active-active",),
        )

    def test_ci_hamiltonians_match_direct_projection(self):
        """Compare the production builder with independent Wannier projection."""
        for name, config in CASES.items():
            with self.subTest(dimension=name):
                if name not in self.references:
                    self.references[name] = build_reference(
                        config["lattice"], config["kmesh"],
                    )
                klas, mo_coeff = self.references[name]
                ci = copy_ci(klas.ci)
                ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
                _, direction = make_ci_direction(ugg, config["seed"] + 100)
                ci = displace_ci(ci, direction, 0.1)
                h1, ecore = klas.h1e_for_cas(mo_coeff=mo_coeff)
                h2 = klas.get_h2cas(mo_coeff)
                solver = make_product_solver(klas)
                projected, _, _ = solver.project_hfrag(
                    h1, h2, [roots[0] for roots in ci],
                    klas.ncas_sub, klas.nelecas_sub, ecore=ecore,
                )
                automatic = klas.h1e_for_las(
                    mo_coeff=mo_coeff, ci=ci, eri_cas=h2,
                )
                np.testing.assert_allclose(
                    np.asarray(automatic)[:, 0], np.asarray(projected),
                    atol=1e-10, rtol=1e-9,
                )

    def test_ci_gradient_forward_difference_order(self):
        """Validate the production automatic builder in every dimension."""
        self.check_convergence(klasscf_ci_grad_fd_check.evaluate)

    def test_orbital_hessian_vector_forward_difference_order(self):
        self.check_convergence(klasscf_orb_hess_fd_check.evaluate, steps=ORBITAL_STEPS)

    def test_ci_hessian_vector_forward_difference_order(self):
        """Compare the production CI Hessian with production gradient differences."""
        self.check_convergence(klasscf_ci_hess_fd_check.evaluate)


if __name__ == "__main__":
    unittest.main()
