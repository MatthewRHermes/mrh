#!/usr/bin/env python

import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.real_linear_solvers import (
    SolveScipyCGForCplx,
    SolveScipyMINRESForCplx,
)


def make_complex_hessian(real_hessian):
    """Return a complex-vector action for a small real Hessian matrix."""
    def hessian_action(vector):
        hessian_action.call_count += 1
        real_vector = SolveScipyCGForCplx.unpack_complex(vector)
        real_result = real_hessian @ real_vector
        return SolveScipyCGForCplx.pack_real(real_result)

    hessian_action.call_count = 0
    return hessian_action


class KnownValuesRealLinearSolvers(unittest.TestCase):

    def test_cg_solves_positive_definite_hessian(self):
        hessian_real = np.array([
            [4.0, 1.0],
            [1.0, 3.0],
        ])
        gradient = np.array([0.5 - 0.8j])
        expected_real = np.linalg.solve(
            hessian_real,
            -SolveScipyCGForCplx.unpack_complex(gradient),
        )

        hessian_action = make_complex_hessian(hessian_real)
        callback_steps = []
        solver = SolveScipyCGForCplx(
            hessian_action,
            real_hdiag=np.diag(hessian_real),
            rtol=1e-12,
            atol=1e-14,
            callback=callback_steps.append,
            compute_residual=True,
        )
        step, info = solver(gradient)

        self.assertEqual(info, 0)
        np.testing.assert_allclose(
            step,
            SolveScipyCGForCplx.pack_real(expected_real),
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertLess(solver.residual_norm, 1e-12)
        self.assertEqual(
            hessian_action.call_count, len(callback_steps) + 1,
        )

    def test_minres_solves_indefinite_hessian(self):
        hessian_real = np.array([
            [2.0, 1.0],
            [1.0, -1.0],
        ])
        gradient = np.array([-0.3 + 0.7j])
        expected_real = np.linalg.solve(
            hessian_real,
            -SolveScipyMINRESForCplx.unpack_complex(gradient),
        )

        hessian_action = make_complex_hessian(hessian_real)
        callback_steps = []
        solver = SolveScipyMINRESForCplx(
            hessian_action,
            rtol=1e-12,
            callback=callback_steps.append,
        )
        step, info = solver(gradient)

        self.assertIsInstance(solver, SolveScipyCGForCplx)
        self.assertEqual(info, 0)
        np.testing.assert_allclose(
            step,
            SolveScipyMINRESForCplx.pack_real(expected_real),
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertIsNone(solver.residual_norm)
        self.assertEqual(
            hessian_action.call_count, len(callback_steps),
        )


if __name__ == "__main__":
    unittest.main()
