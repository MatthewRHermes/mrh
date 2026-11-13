#!/usr/bin/env python
# LASPSCF kernel started from fragment 1-RDMs (casdm0_frs, used by my_dmet) instead of CI
# vectors. The other tests that pass casdm0_frs (test_old_*) converge to the same answer from any
# starting point, so they would not notice if the starting point were wrong.

import unittest
import numpy as np
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.laspscf import LASPSCF

def setUpModule():
    global mf
    mol = gto.M (atom='H 0 0 0; H 0.74 0 0; H 0.3 2.0 0; H 1.2 2.3 0', basis='6-31g',
                 symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()

def tearDownModule():
    global mf
    mf.mol.stdout.close ()
    del mf

class _Stop (Exception):
    pass

def first_veff (las, mo, **kwargs):
    '''The effective potential, and its 1-RDMs, of the kernel's first macrocycle'''
    out = {}
    get_veff = las.get_veff
    def stop (dm=None, **kw):
        out['veff'] = get_veff (dm=dm, **kw)
        out['dm1s'] = dm
        if las._casdm0_frs is not None: out['casdm1frs'] = las._starting_casdm1frs (None)
        raise _Stop
    las.get_veff = stop
    try:
        las.kernel (mo, **kwargs)
    except _Stop:
        pass
    finally:
        del las.get_veff
    return out

class KnownValues (unittest.TestCase):

    def test_starting_point (self):
        # Starting from the 1-RDMs of some CI vectors gives the same potential as starting from
        # the CI vectors themselves
        las = LASPSCF (mf, (2,2), (2,2))
        mo = las.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)
        las.lasci (mo)
        ci = las.ci
        casdm0_frs = las.states_make_casdm1s_sub (ci=ci)
        las.ci = None
        test = first_veff (las, mo, casdm0_frs=casdm0_frs)
        self.assertIsNone (las.ci)
        self.assertIs (test['casdm1frs'], casdm0_frs)
        las.ci = ci
        ref = first_veff (las, mo)
        self.assertAlmostEqual (np.abs (test['dm1s'] - ref['dm1s']).max (), 0, 10)
        self.assertAlmostEqual (np.abs (test['veff'] - ref['veff']).max (), 0, 10)
        self.assertIsNone (las._casdm0_frs)

    def test_ci0_wins (self):
        # Given guess CI vectors, the kernel ignores casdm0_frs
        las = LASPSCF (mf, (2,2), (2,2))
        mo = las.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)
        las.lasci (mo)
        casdm0_frs = [np.zeros_like (dm) for dm in las.states_make_casdm1s_sub ()]
        test = first_veff (las, mo, casdm0_frs=casdm0_frs)
        ref = first_veff (las, mo)
        self.assertAlmostEqual (np.abs (test['veff'] - ref['veff']).max (), 0, 10)

    def test_kernel (self):
        las_ref = LASPSCF (mf, (2,2), (2,2))
        mo = las_ref.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)
        las_ref.kernel (mo)
        self.assertTrue (las_ref.converged)
        casdm0_frs = las_ref.states_make_casdm1s_sub ()
        las = LASPSCF (mf, (2,2), (2,2))
        las.kernel (mo, casdm0_frs=casdm0_frs)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, las_ref.e_tot, 7)
        # The kernel doesn't keep casdm0_frs around for the next call
        self.assertIsNone (las._casdm0_frs)

if __name__ == "__main__":
    print("Full Tests for LASPSCF starting from casdm0_frs")
    unittest.main()
