#!/usr/bin/env python

"""Check k-LASSCF optimization, residual stopping, trust limits, energy acceptance,
fragment CI solves, and energy normalization, with periodic integration checks."""

import sys
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas, klasscf


class KnownValuesKLASSCFOptimizerResults(unittest.TestCase):

    def test_kernel_method_stores_optimizer_results(self):
        class FakeOptimizer:
            mo_coeff = np.array([[[1.0]]])
            ci = [[np.array([1.0])]]
            verbose = lib.logger.QUIET
            conv_tol_grad = 1e-7
            conv_tol = 1e-7
            sanity_calls = 0
            flag_calls = []
            finalize_calls = []

            def check_sanity(self):
                self.sanity_calls += 1

            def dump_flags(self, verbose):
                self.flag_calls.append(verbose)

            def _finalize(self, method=None):
                self.finalize_calls.append(method)

        optimizer = FakeOptimizer()
        initial_mo = optimizer.mo_coeff
        final_mo = np.array([[[2.0]]])
        final_ci = [[np.array([0.5])]]
        expected = (
            True, -1.2, np.array([-1.2]), np.array([[0.3]]), final_mo,
            np.array([-0.8]), [[np.array([0.0])]], final_ci,
            np.ones((1, 1, 1, 1)), np.ones((2, 1, 1, 1)),
        )
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return expected

        actual = klasscf._klasscf_kernel_method(
            optimizer, _kern=fake_kernel,
        )

        self.assertEqual(calls[0]["conv_tol"], 1e-7)
        self.assertIs(calls[0]["mo_coeff"], initial_mo)
        self.assertTrue(optimizer.converged)
        np.testing.assert_allclose(optimizer.e_tot, -1.2)
        self.assertIs(optimizer.mo_coeff, final_mo)
        self.assertIs(optimizer.ci, final_ci)
        self.assertEqual(optimizer.finalize_calls, ["k-LASSCF"])
        self.assertEqual(len(actual), 7)


class FakeFCIBox:
    def __init__(self, energy):
        self.energy = energy
        self.calls = []

    def kernel(self, h1, h2, norb, nelec, **kwargs):
        self.calls.append((h1, h2, norb, nelec, kwargs))
        return self.energy, kwargs["ci0"]


class KnownValuesKLASSCFKeyframe(unittest.TestCase):

    def test_ci_cycle_solves_each_unfrozen_fragment_once(self):
        boxes = [FakeFCIBox(-0.4), FakeFCIBox(-0.3)]

        class FakeLAS:
            fciboxes = boxes
            ncas_sub = np.array([1, 1])
            nelecas_sub = np.array([(1, 0), (0, 1)])
            frozen_ci = None
            max_memory = 1000

            def h1e_for_las(self, **kwargs):
                return [
                    np.full((1, 2, 1, 1), 0.2),
                    np.full((1, 2, 1, 1), 0.3),
                ]

        ci0 = [[np.array([1.0])], [np.array([1.0])]]
        h2eff = np.arange(16.0).reshape((2, 2, 2, 2))
        energies, ci1 = klasscf.ci_cycle(
            FakeLAS(), np.ones((1, 1, 1)), ci0,
            np.ones((2, 1, 1, 1)), h2eff,
            [np.ones((1, 2, 1, 1))] * 2,
            lib.logger.Logger(sys.stdout, lib.logger.QUIET),
        )

        np.testing.assert_allclose(energies, [-0.4, -0.3])
        self.assertEqual(ci1, ci0)
        self.assertEqual([len(box.calls) for box in boxes], [1, 1])
        np.testing.assert_allclose(
            boxes[0].calls[0][1], h2eff[:1, :1, :1, :1],
        )
        np.testing.assert_allclose(
            boxes[1].calls[0][1], h2eff[1:, 1:, 1:, 1:],
        )

    def test_ci_cycle_preserves_frozen_fragment(self):
        boxes = [FakeFCIBox(-0.4), FakeFCIBox(-0.3)]

        class FakeLAS:
            fciboxes = boxes
            ncas_sub = np.array([1, 1])
            nelecas_sub = np.array([(1, 0), (0, 1)])
            frozen_ci = [1]
            max_memory = 1000

            def h1e_for_las(self, **kwargs):
                return [np.zeros((1, 2, 1, 1))] * 2

        ci0 = [[np.array([1.0])], [np.array([2.0])]]
        energies, ci1 = klasscf.ci_cycle(
            FakeLAS(), np.ones((1, 1, 1)), ci0,
            np.ones((2, 1, 1, 1)), np.zeros((2, 2, 2, 2)),
            [np.ones((1, 2, 1, 1))] * 2,
            lib.logger.Logger(sys.stdout, lib.logger.QUIET),
        )

        np.testing.assert_allclose(energies, [-0.4, 0.0])
        self.assertEqual([len(box.calls) for box in boxes], [1, 0])
        self.assertIs(ci1[1], ci0[1])

    def test_fixed_ci_energies_preserve_root_and_cell_normalization(self):
        boxes = [
            SimpleNamespace(fcisolvers=["a0", "a1"]),
            SimpleNamespace(fcisolvers=["b0", "b1"]),
        ]

        class FakeLAS:
            nroots = 2
            nfrags = 2
            nkpts = 2
            ncas = 2
            ncore = 0
            ncas_sub = [1, 1]
            nelecas_sub = [(1, 0), (0, 1)]
            weights = np.array([0.25, 0.75])
            fciboxes = boxes
            stdout = sys.stdout

            def h1e_for_cas(self, **kwargs):
                return np.ones((2, 2)), 4.0

        ci = [
            [np.array([1.0]), np.array([2.0])],
            [np.array([3.0]), np.array([4.0])],
        ]
        active_energies = iter([2.0, 6.0])
        solver_calls = []

        class FakeProductSolver:
            def __init__(self, fcisolvers, **kwargs):
                solver_calls.append((fcisolvers, kwargs))

            def energy_elec(self, *args, **kwargs):
                return next(active_energies)

        with patch.object(
                klasscf, "ImpureProductStateFCISolver", FakeProductSolver):
            e_tot, e_states, e_cas, e_lexc = klasscf._fixed_ci_energies(
                FakeLAS(), np.ones((1, 2, 2)), ci,
                np.ones((2, 2, 2, 2)),
            )

        np.testing.assert_allclose(e_cas, [1.0, 3.0])
        np.testing.assert_allclose(e_states, [3.0, 5.0])
        np.testing.assert_allclose(e_tot, 4.5)
        self.assertEqual(solver_calls[0][0], ["a0", "b0"])
        self.assertEqual(solver_calls[1][0], ["a1", "b1"])
        self.assertEqual(len(e_lexc), 2)
        self.assertEqual([len(roots) for roots in e_lexc], [2, 2])

    def test_mo_energies_are_spin_averaged_fock_diagonals(self):
        h1s = np.array([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ])
        actual = klasscf._get_mo_energy(SimpleNamespace(h1s=h1s))
        np.testing.assert_allclose(actual, [[3.0, 6.0]])
        self.assertIsNone(
            klasscf._get_mo_energy(SimpleNamespace(h1s=None))
        )


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
        # These step/acceptance tests isolate optimizer mechanics from the
        # stricter physical energy stopping tolerance tested separately.
        self.conv_tol = 0.2
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

    def test_energy_tolerance_requires_a_small_macro_change(self):
        for tolerance, expected in [(1e-7, True), (1e-8, False)]:
            with self.subTest(conv_tol=tolerance):
                first = FakeHop([.4])
                stationary = FakeHop([0.0])
                las = FakeKLASSCF([first, stationary, stationary])
                las.max_cycle_macro = 2
                with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                        patch.object(klasscf, "_fixed_ci_energies", side_effect=[
                            fixed_energy(-1.0), fixed_energy(-1.1),
                            fixed_energy(-1.1), fixed_energy(-1.10000005),
                        ]):
                    result = klasscf.kernel(las, conv_tol=tolerance)
                self.assertEqual(result[0], expected)
                self.assertEqual(len(las.uggs), 3)
                self.assertEqual(len(first.steps), 1)
                self.assertEqual(len(stationary.steps), 0)

    def test_initial_keyframe_is_not_energy_convergence(self):
        las = FakeKLASSCF([FakeHop([0.0])])
        las.conv_tol = 1e-7
        las.max_cycle_macro = 0
        las.verbose = lib.logger.INFO
        las.stdout = io.StringIO()
        with patch.object(klasscf, "ci_cycle", return_value=ci_cycle_result()), \
                patch.object(klasscf, "_fixed_ci_energies", return_value=fixed_energy(-1.0)):
            result = klasscf.kernel(las)
        self.assertFalse(result[0])
        self.assertIn("dE = N/A", las.stdout.getvalue())

    def test_macro_driver_applies_the_complex_newton_step(self):
        first_hop = FakeHop([0.4 + 0.2j], curvature=2.0)
        final_hop = FakeHop([0.0j], curvature=2.0)
        las = FakeKLASSCF([first_hop, final_hop])

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


class KnownValuesKLASSCFOptimizerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.diag([4.0, 10.0, 10.0])
        cell.atom = "H 0 0 0; H 1.5 0 0"
        cell.basis = "sto-3g"
        cell.unit = "Angstrom"
        cell.precision = 1e-10
        cell.ke_cutoff = 20
        cell.verbose = lib.logger.QUIET
        cell.build()

        kmesh = (2, 1, 1)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.exxdiv = None
        kmf.max_cycle = 0
        kmf.kernel()

        mo_avas = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]
        las = mcscf.KLASSCF(
            kmf, 2, (1, 1), kmesh=kmesh, trans_sym=False,
        )
        mo_guess = las.localize_init_guess(
            ["H 1s"], mo_coeff=mo_avas,
        )
        cls.las = las
        cls.mo_guess = mo_guess

    def test_public_optimizer_builds_physical_keyframe(self):
        self.las.max_cycle_macro = 0
        result = self.las.kernel(mo_coeff=self.mo_guess)
        e_tot, e_cas, ci, mo_coeff, mo_energy, h2eff, veff = result

        self.assertIsInstance(self.las, klasscf.PBCLASSCFNoSymm)
        self.assertTrue(np.isfinite(e_tot))
        self.assertTrue(np.all(np.isfinite(e_cas)))
        self.assertEqual(len(ci), self.las.nfrags)
        self.assertEqual(np.shape(mo_coeff), np.shape(self.mo_guess))
        self.assertEqual(np.shape(mo_energy), (
            self.las.nkpts, mo_coeff.shape[-1],
        ))
        ncastot = int(np.sum(self.las.ncas_sub))
        self.assertEqual(np.shape(h2eff), (ncastot,) * 4)
        self.assertEqual(np.shape(veff), (
            2, self.las.nkpts, mo_coeff.shape[-1], mo_coeff.shape[-1],
        ))

    def test_physical_optimizer_applies_one_minres_microiteration(self):
        angles = (0.04, -0.04)
        mo_start = np.asarray([
            mo @ linalg.expm(np.array([
                [0.0, -angle], [angle, 0.0],
            ]))
            for mo, angle in zip(self.mo_guess, angles)
        ])
        self.las.max_cycle_macro = 1
        self.las.max_cycle_micro = 1
        # Force one update even if the initial gradient is already below the
        # usual threshold; the second keyframe is still the stopping point.
        self.las.min_cycle_macro = 2

        result = self.las.kernel(mo_coeff=mo_start)

        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[1]))
        self.assertTrue(np.all(np.isfinite(result[3])))
        self.assertFalse(np.allclose(result[3], mo_start))


if __name__ == "__main__":
    unittest.main()
