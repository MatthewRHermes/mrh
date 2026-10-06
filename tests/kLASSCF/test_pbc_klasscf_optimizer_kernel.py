#!/usr/bin/env python

import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import klasscf


class FakeUGG:
    nvar_orb = 1
    nvar_tot = 1


class FakeHop:
    def __init__(self, gradient, curvature=2.0):
        self.gradient = np.asarray(gradient, dtype=np.complex128)
        self.curvature = curvature
        self.steps = []
        self.h1s = np.ones((2, 1, 1, 1))

    def get_grad(self):
        return self.gradient

    def _matvec(self, vector):
        return self.curvature * np.asarray(vector)

    def update_mo_ci_eri(self, step, h2eff):
        self.steps.append(np.array(step, copy=True))
        return (
            np.array([[[step[0]]]]),
            [[np.array([1.0 + 0.0j])]],
            np.asarray(h2eff) + 1.0,
        )


class FakeKLASSCF:
    def __init__(self, hops, trust_radius=10.0):
        self.mo_coeff = np.zeros((1, 1, 1), dtype=np.complex128)
        self.ci = [[np.array([1.0 + 0.0j])]]
        self.conv_tol_grad = 1e-9
        self.max_cycle_macro = 1
        self.max_cycle_micro = 5
        self.min_cycle_macro = 0
        self.trust_radius = trust_radius
        self.weights = [1.0]
        self.nroots = 1
        self.nfrags = 1
        self.nkpts = 1
        self.ncas = 1
        self.ncore = 0
        self.verbose = lib.logger.QUIET
        self.stdout = sys.stdout
        self._scf = SimpleNamespace(cell=object())
        self.hops = list(hops)
        self.uggs = []
        self.hop_kwargs = []

    def get_h2cas(self, mo_coeff):
        return np.full((1, 1, 1, 1), 3.0)

    def states_make_casdm1s_sub(self, ci=None):
        return [np.ones((1, 2, 1, 1))]

    def make_casdm1s_sub(self, ci=None, casdm1frs=None):
        return [np.ones((2, 1, 1))]

    def make_rdm1s(self, mo_coeff=None, ci=None, casdm1s_sub=None):
        return np.ones((2, 1, 1, 1))

    def get_veff(self, cell, dm_kpts=None):
        return np.ones((2, 1, 1, 1))

    def get_ugg(self, mo_coeff=None, ci=None):
        ugg = FakeUGG()
        self.uggs.append(ugg)
        return ugg

    def get_hop(self, mo_coeff=None, ci=None, ugg=None, **kwargs):
        self.hop_kwargs.append(kwargs)
        return self.hops.pop(0)


def ci_cycle_result():
    return [0.0], [[np.array([1.0 + 0.0j])]]


def fixed_energy(energy):
    return (
        energy, np.array([energy]), np.array([energy - 0.5]),
        [[np.array([0.0])]],
    )


class KnownValuesKLASSCFKernel(unittest.TestCase):

    def test_macro_driver_applies_the_complex_newton_step(self):
        first_hop = FakeHop([0.4 + 0.2j], curvature=2.0)
        final_hop = FakeHop([0.0j], curvature=2.0)
        las = FakeKLASSCF([first_hop, final_hop])
        las.verbose = lib.logger.INFO
        las.stdout = io.StringIO()

        with patch.object(
                klasscf, "ci_cycle",
                side_effect=[ci_cycle_result(), ci_cycle_result()]), \
                patch.object(
                    klasscf, "_fixed_ci_energies",
                    side_effect=[fixed_energy(-1.0), fixed_energy(-1.1), fixed_energy(-1.1)],
                ):
            result = klasscf.kernel(las)

        self.assertTrue(result[0])
        self.assertEqual(len(first_hop.steps), 1)
        np.testing.assert_allclose(
            first_hop.steps[0], np.array([-0.2 - 0.1j]), atol=1e-12,
        )
        np.testing.assert_allclose(result[1], -1.1)
        self.assertEqual(len(las.uggs), 2)
        self.assertIsNot(las.uggs[0], las.uggs[1])
        np.testing.assert_allclose(las.hop_kwargs[0]["h2eff"], 3.0)
        np.testing.assert_allclose(las.hop_kwargs[1]["h2eff"], 4.0)
        self.assertIn("micro iter 0 : |r_orb| =", las.stdout.getvalue())
        self.assertIn("|r_ci| =", las.stdout.getvalue())
        self.assertIn("Accepted k-LASSCF trial:", las.stdout.getvalue())

    def test_soft_mode_step_can_grow_and_stop_on_actual_residual(self):
        class TwoVariableUGG:
            nvar_orb, nvar_tot = 1, 2

        class SoftHop(FakeHop):
            def _matvec(self, vector):
                return np.array([1., .001]) * vector

        solver_options = []

        class CapturingMINRES(klasscf.SolveScipyMINRESForCplx):
            def __init__(self, *args, **kwargs):
                solver_options.append(kwargs)
                super().__init__(*args, **kwargs)

        first_hop = SoftHop([1e-4, 1e-4])
        las = FakeKLASSCF([first_hop, SoftHop([0., 0.])])
        las.conv_tol_grad = 1e-4
        las.max_cycle_micro_near_convergence = 17
        las.micro_solver = CapturingMINRES
        las.verbose, las.stdout = lib.logger.INFO, io.StringIO()

        with patch.object(las, "get_ugg", return_value=TwoVariableUGG()), \
                patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                    fixed_energy(-1.), fixed_energy(-1.1), fixed_energy(-1.1),
                ]):
            result = klasscf.kernel(las)

        self.assertTrue(result[0])
        self.assertEqual(solver_options[0]["maxiter"], 17)
        np.testing.assert_allclose(first_hop.steps[0], [-1e-4, -.1], atol=1e-12)
        np.testing.assert_allclose(
            first_hop.gradient + first_hop._matvec(first_hop.steps[0]), 0., atol=1e-12,
        )
        self.assertNotIn("Unstable", las.stdout.getvalue())

    def test_later_iterate_with_worse_residual_keeps_better_step(self):
        class WorseningSolver:
            def __init__(self, matvec, callback, **kwargs):
                self.callback = callback

            def __call__(self, gradient, **kwargs):
                self.callback(np.array([-.5]))
                self.callback(np.array([-.4]))
                return np.array([-.4]), 5

        hop = FakeHop([1.], curvature=1.)
        las = FakeKLASSCF([hop, FakeHop([0.])])
        las.micro_solver = WorseningSolver
        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                    fixed_energy(-1.), fixed_energy(-1.1), fixed_energy(-1.1),
                ]):
            result = klasscf.kernel(las)
        self.assertTrue(result[0])
        np.testing.assert_allclose(hop.steps[0], [-.5])

    def test_cached_hessian_action_respects_real_complex_coordinates(self):
        basis = [np.array([1., 0.]), np.array([1.j, 0.])]
        actions = [np.array([2., 0.]), np.array([3.j, 0.])]
        unused = Mock()
        action = klasscf._micro_hessian_action(
            np.array([.4 + .2j, 0.]), basis, actions, unused,
        )
        np.testing.assert_allclose(action, [.8 + .6j, 0.])
        unused.assert_not_called()
        fallback = Mock(return_value=np.array([0., 5.]))
        action = klasscf._micro_hessian_action(
            np.array([0., 1.]), basis, actions, fallback,
        )
        np.testing.assert_allclose(action, [0., 5.])
        fallback.assert_called_once()

    def test_no_residual_improvement_recovers_a_projected_newton_step(self):
        class PoorSolver:
            def __init__(self, matvec, callback, **kwargs):
                self.matvec = matvec
                self.callback = callback

            def __call__(self, gradient, **kwargs):
                self.matvec(np.array([1.]))
                self.callback(np.array([-1.4]))
                return np.array([-1.4]), 5

        hop = FakeHop([1.], curvature=2.)
        las = FakeKLASSCF([hop, FakeHop([0.])])
        las.micro_solver = PoorSolver
        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                    fixed_energy(-1.), fixed_energy(-1.1), fixed_energy(-1.1),
                ]):
            result = klasscf.kernel(las)
        self.assertTrue(result[0])
        np.testing.assert_allclose(hop.steps[0], [-.5])

    def test_nonfinite_solver_result_uses_a_finite_fallback(self):
        class NonfiniteSolver:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return np.array([np.nan]), 5

        hop = FakeHop([1.])
        las = FakeKLASSCF([hop, FakeHop([0.])])
        las.micro_solver = NonfiniteSolver
        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                    fixed_energy(-1.), fixed_energy(-1.1), fixed_energy(-1.1),
                ]):
            result = klasscf.kernel(las)
        self.assertTrue(result[0])
        np.testing.assert_allclose(hop.steps[0], [-1.])

    def test_macro_driver_limits_a_large_step_to_the_trust_radius(self):
        first_hop = FakeHop([10.0 + 0.0j], curvature=1.0)
        final_hop = FakeHop([1.0 + 0.0j], curvature=1.0)
        las = FakeKLASSCF([first_hop, final_hop], trust_radius=0.25)

        with patch.object(
                klasscf, "ci_cycle",
                side_effect=[ci_cycle_result(), ci_cycle_result()]), \
                patch.object(
                    klasscf, "_fixed_ci_energies",
                    side_effect=[fixed_energy(-1.0), fixed_energy(-1.05), fixed_energy(-1.05)],
                ):
            result = klasscf.kernel(las)

        self.assertFalse(result[0])
        self.assertEqual(len(first_hop.steps), 1)
        np.testing.assert_allclose(first_hop.steps[0], [-0.25 + 0.0j])

    def test_macro_driver_rejects_state_averaged_metric(self):
        las = FakeKLASSCF([FakeHop([0.1 + 0.0j])])
        las.weights = [0.5, 0.5]
        las.nroots = 2
        with patch.object(
                klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(
                    klasscf, "_fixed_ci_energies",
                    return_value=fixed_energy(-1.0),
                ):
            with self.assertRaisesRegex(NotImplementedError, "state-averaged"):
                klasscf.kernel(las)

    def test_uphill_newton_step_is_halved_until_actual_energy_decreases(self):
        first_hop = FakeHop([1.0], curvature=1.0)
        las = FakeKLASSCF([first_hop, FakeHop([0.0])])

        def actual_energy(las, mo, ci, h2):
            x = mo[0, 0, 0].real
            return fixed_energy(x + 2 * x*x)

        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=actual_energy):
            result = klasscf.kernel(las)

        self.assertTrue(result[0])
        np.testing.assert_allclose([s[0] for s in first_hop.steps], [-1, -.5, -.25])
        np.testing.assert_allclose(result[4], [[[-.25]]])
        self.assertAlmostEqual(result[1], -.125)

    def test_failed_backtracking_preserves_last_accepted_state(self):
        hop = FakeHop([1.0])
        las = FakeKLASSCF([hop])
        las.max_step_backtracks = 2
        initial_mo = las.mo_coeff.copy()

        def uphill_energy(las, mo, ci, h2):
            return fixed_energy(abs(mo[0, 0, 0]))

        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=uphill_energy):
            result = klasscf.kernel(las)

        self.assertFalse(result[0])
        self.assertEqual(len(hop.steps), 6)
        np.testing.assert_array_equal(result[4], initial_mo)
        np.testing.assert_array_equal(result[8], np.full((1, 1, 1, 1), 3.0))
        self.assertEqual(result[1], 0.0)

    def test_negative_gradient_fallback_after_failed_newton_trial(self):
        las = FakeKLASSCF([])
        hop = FakeHop([1.0])

        def actual_energy(las, mo, ci, h2):
            x = mo[0, 0, 0].real
            return fixed_energy(x + .75 * x*x)

        with patch.object(klasscf, "_fixed_ci_energies", side_effect=actual_energy):
            accepted = klasscf._backtrack_macro_step(
                las, hop, np.array([-1.5]), np.array([1.0]),
                np.full((1, 1, 1, 1), 3.0), 0.0, 10.0, 0,
                lib.logger.Logger(sys.stdout, lib.logger.QUIET),
            )
        self.assertIsNotNone(accepted)
        np.testing.assert_allclose(accepted[0], [[[-1.0]]])
        self.assertAlmostEqual(accepted[3][0], -.25)

    def test_uphill_synchronous_ci_refresh_preserves_accepted_trial(self):
        las = FakeKLASSCF([FakeHop([.4]), FakeHop([0.0])])
        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                    fixed_energy(-1.0), fixed_energy(-1.1), fixed_energy(-1.05),
                ]):
            result = klasscf.kernel(las)
        self.assertTrue(result[0])
        self.assertEqual(result[1], -1.1)


if __name__ == "__main__":
    unittest.main()
