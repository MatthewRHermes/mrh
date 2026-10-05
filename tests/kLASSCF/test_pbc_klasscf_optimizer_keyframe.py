#!/usr/bin/env python

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import klasscf


class FakeFCIBox:
    def __init__(self, energy):
        self.energy = energy
        self.calls = []

    def kernel(self, h1, h2, norb, nelec, **kwargs):
        self.calls.append((h1, h2, norb, nelec, kwargs))
        return self.energy, kwargs["ci0"]


class KnownValuesKLASSCFKeyframe(unittest.TestCase):

    def test_optimizer_metric_accepts_only_one_unit_weight(self):
        ugg = SimpleNamespace(nvar_tot=3)
        metric = klasscf._optimizer_metric(
            SimpleNamespace(weights=[1.0], nroots=1), ugg,
        )
        np.testing.assert_allclose(metric, np.ones(3))
        with self.assertRaisesRegex(NotImplementedError, "state-averaged"):
            klasscf._optimizer_metric(
                SimpleNamespace(weights=[0.5, 0.5], nroots=2), ugg,
            )
        with self.assertRaisesRegex(ValueError, "weights has size"):
            klasscf._optimizer_metric(
                SimpleNamespace(weights=[1.0], nroots=2), ugg,
            )

    def test_missing_ci_guess_detects_nested_none(self):
        self.assertTrue(klasscf._ci_guess_is_missing(None))
        self.assertTrue(klasscf._ci_guess_is_missing([None]))
        self.assertTrue(klasscf._ci_guess_is_missing([[None]]))
        self.assertFalse(
            klasscf._ci_guess_is_missing([[np.array([1.0])]])
        )

    def test_keyframe_densities_follow_periodic_las_pipeline(self):
        calls = []
        casdm1frs = [np.ones((1, 2, 1, 1))]
        casdm1s_sub = [np.ones((2, 1, 1))]
        dm1s_kpts = np.ones((2, 1, 1, 1))
        veff_kpts = np.full((2, 1, 1, 1), 2.0)

        class FakeLAS:
            _scf = SimpleNamespace(cell=object())

            def states_make_casdm1s_sub(self, ci=None):
                calls.append(("states", ci))
                return casdm1frs

            def make_casdm1s_sub(self, ci=None, casdm1frs=None):
                calls.append(("active", ci, casdm1frs))
                return casdm1s_sub

            def make_rdm1s(self, **kwargs):
                calls.append(("ao", kwargs))
                return dm1s_kpts

            def get_veff(self, cell, dm_kpts=None):
                calls.append(("veff", cell, dm_kpts))
                return veff_kpts

        mo = np.ones((1, 1, 1))
        ci = [[np.array([1.0])]]
        actual = klasscf._make_keyframe_densities(FakeLAS(), mo, ci)

        self.assertIs(actual[0], casdm1frs)
        self.assertIs(actual[1], casdm1s_sub)
        self.assertIs(actual[2], dm1s_kpts)
        self.assertIs(actual[3], veff_kpts)
        self.assertEqual([call[0] for call in calls], [
            "states", "active", "ao", "veff",
        ])

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


if __name__ == "__main__":
    unittest.main()
