import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf 
from mrh.my_pyscf import mcpdft
import unittest, math

degree = math.pi / 180.0
theta0 = 30
def u_theta (theta=theta0):
    # The sign here is consistent with my desired variable convention:
    # lower-triangular positive w/ row idx = initial state and col idx = 
    # final state
    ct = math.cos (theta*degree)
    st = math.sin (theta*degree)
    return np.array ([[ct,st],[-st,ct]])
def numerical_Q (mf, mc):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    mo_cas = mc.mo_coeff[:,ncore:][:,:ncas]
    def num_Q (theta):
        ci_theta = mc.get_ci_basis (uci=u_theta(theta))
        states_casdm1 = mc.fcisolver.states_make_rdm1 (
            ci_theta, ncas, nelecas)
        Q = 0
        for casdm1 in states_casdm1:
            dm_cas = np.dot (mo_cas, casdm1)
            dm_cas = np.dot (dm_cas, mo_cas.conj ().T)
            vj = mf.get_j (dm=dm_cas)
            Q += np.dot (vj.ravel (), dm_cas.ravel ())/2
        return Q
    return num_Q

def get_lih (r):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    mc = mc.state_interaction ([0.5,0.5], 'cms').run (conv_tol=1e-8)
    return mol, mf, mc

def setUpModule():
    global mol, mf, mc
    mol, mf, mc = get_lih (1.5)

def tearDownModule():
    global mol, mf, mc
    mol.stdout.close ()
    del mol, mf, mc

class KnownValues(unittest.TestCase):

    def test_derivs (self):
        Q_max = mc.diabatizer ()[0]
        ci_theta0 = mc.get_ci_basis (uci=u_theta(theta0))
        Q_test, dQ_test, d2Q_test = mc.diabatizer (ci=ci_theta0)[:3]
        num_Q = numerical_Q (mf, mc)
        delta=0.01
        Qm = num_Q (theta0-delta)
        Q0 = num_Q (theta0)
        Qp = num_Q (theta0+delta)
        dQ_ref = (Qp-Qm)/2/delta/degree
        d2Q_ref = (Qp+Qm-2*Q0)/degree/degree/delta/delta
        with self.subTest (deriv=0):
            self.assertLess (Q_test, Q_max)
            self.assertAlmostEqual (Q_test, Q0, 9)
        with self.subTest (deriv=1):
            self.assertAlmostEqual (dQ_test[0], dQ_ref, 6)
        with self.subTest (deriv=2):
            self.assertAlmostEqual (d2Q_test[0,0], d2Q_ref, 6)

    def test_old_implementation (self):
        ci_theta0 = mc.get_ci_basis (uci=u_theta(theta0))
        Q_test, dQ_test, d2Q_test = mc.diabatizer (ci=ci_theta0)[:3]
        from mrh.my_pyscf.mcpdft.cmspdft import e_coul_o0
        Q_ref, dQ_ref, d2Q_ref = e_coul_o0 (mc, ci_theta0)
        with self.subTest (deriv=0):
            self.assertAlmostEqual (Q_test, Q_ref, 9)
        with self.subTest (deriv=1):
            self.assertAlmostEqual (dQ_test[0], dQ_ref[0], 9)
        with self.subTest (deriv=2):
            self.assertAlmostEqual (d2Q_test[0,0], d2Q_ref[0,0], 9)
    

if __name__ == "__main__":
    print("Full Tests for CMS-PDFT objective function")
    unittest.main()






