import unittest
import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import get_grad_orb
from mrh.my_pyscf.mcscf.lasscf_async_split import LASImpurityOrbitalCallable
from mrh.my_pyscf.mcscf.lasscf_async_crunch import get_impurity_casscf

def setUpModule():
    global las, fo_coeff, nelec_fo
    xyz='''Li 0 0 0,
           H 2 0 0,
           Li 10 0 0,
           H 12 0 0'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    mo = las.localize_init_guess (([0,1],[2,3]), mc.mo_coeff, freeze_cas_spaces=True)
    las.kernel (mo)
    las.state_average_(weights=[.2,.2,.2,.2,.2],
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    las.lasci ()
    las.conv_tol_grad = 1e-6
    las.kernel ()
    assert (las.converged)

    ###########################
    # Build the embedding space
    dm1s = las.make_rdm1s ()
    veff = las.get_veff (dm1s=dm1s, spin_sep=True)
    fock1 = get_grad_orb (las, hermi=0)
    get_imporbs_0 = LASImpurityOrbitalCallable (las, 0, list (range (3)))
    fo_coeff, nelec_fo = get_imporbs_0 (las.mo_coeff, dm1s, veff, fock1)
    ###########################
    
    ###########################
    # Build the impurity method object
    imc = get_impurity_casscf (las, 0)
    imc._update_space_(fo_coeff, nelec_fo)
    imc._update_keyframe_(las.mo_coeff, las.ci)
    ###########################

def tearDownModule():
    global las, fo_coeff, nelec_fo
    las.stdout.close ()
    del las, fo_coeff, nelec_fo

def _make_imc (kv):
    imc = get_impurity_casscf (las, 0)
    imc._update_space_(fo_coeff, nelec_fo)
    imc._update_keyframe_(las.mo_coeff, las.ci)
    imc.conv_tol = 1e-10
    imc.kernel ()
    with kv.subTest ('impurity CASSCF converged'):
        kv.assertTrue (imc.converged)
    return imc

def _test_energies (kv, imc, tag):
    with kv.subTest (tag + ' state-averaged energy'):
        kv.assertAlmostEqual (imc.e_tot, las.e_tot, 8)
    for i, (t, r) in enumerate (zip (imc.e_states, las.e_states)):
        with kv.subTest (tag, state=i):
            kv.assertAlmostEqual (t, r, 6)

def _perturb_wfn (imc):
    imc.ci = None
    kappa = (np.random.rand (*imc.mo_coeff.shape)-.5) * np.pi / 100
    kappa -= kappa.T
    umat = linalg.expm (kappa)
    imc.mo_coeff = imc.mo_coeff @ umat
    return imc

class KnownValues (unittest.TestCase):

    def test_energies_and_optimization (self):
        imc = _make_imc (self)
        _test_energies (self, imc, 'energy')
        imc = _perturb_wfn (imc)
        _test_energies (self, imc, 'optimization')

if __name__ == "__main__":
    print("Full Tests for lasscf_async_crunch")
    unittest.main()
