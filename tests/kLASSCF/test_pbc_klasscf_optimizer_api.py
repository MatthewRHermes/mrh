#!/usr/bin/env python

import unittest
from unittest.mock import patch

import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.klasci import PBCLASCINoSymm


class KnownValuesKLASSCFAPI(unittest.TestCase):

    def test_optimizer_class_selects_general_hessian_and_minres(self):
        self.assertTrue(issubclass(
            klasscf.PBCLASSCFNoSymm, PBCLASCINoSymm,
        ))
        self.assertIs(
            klasscf.PBCLASSCFNoSymm._hop,
            klasscf.KLASSCF_HessianOperator,
        )
        self.assertIs(
            klasscf.PBCLASSCFNoSymm.micro_solver,
            klasscf.SolveScipyMINRESForCplx,
        )

    def test_kernel_method_stores_optimizer_results(self):
        class FakeOptimizer:
            mo_coeff = np.array([[[1.0]]])
            ci = [[np.array([1.0])]]
            verbose = lib.logger.QUIET
            conv_tol_grad = 1e-7
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

        self.assertIs(calls[0]["mo_coeff"], initial_mo)
        self.assertTrue(optimizer.converged)
        np.testing.assert_allclose(optimizer.e_tot, -1.2)
        self.assertIs(optimizer.mo_coeff, final_mo)
        self.assertIs(optimizer.ci, final_ci)
        self.assertEqual(optimizer.finalize_calls, ["LASSCF"])
        self.assertEqual(len(actual), 7)

    def test_factory_promotes_general_klasci_object(self):
        base = PBCLASCINoSymm.__new__(PBCLASCINoSymm)
        with patch.object(klasscf, "kLASCI", return_value=base) as factory:
            actual = klasscf.kLASSCF(
                object(), 2, (1, 1), kmesh=(2, 1, 1),
            )

        self.assertIs(actual, base)
        self.assertIsInstance(actual, klasscf.PBCLASSCFNoSymm)
        self.assertFalse(factory.call_args.kwargs["trans_sym"])

    def test_factory_rejects_translation_adapted_optimizer(self):
        with self.assertRaisesRegex(
                NotImplementedError, "translation-adapted"):
            klasscf.kLASSCF(
                object(), 2, (1, 1), kmesh=(2, 1, 1), trans_sym=True,
            )

    def test_public_alias_uses_optimizer_factory(self):
        self.assertIs(mcscf.KLASSCF, klasscf.kLASSCF)


if __name__ == "__main__":
    unittest.main()
