import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib
from mrh.my_pyscf.lassi.lassis.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.lassis.min_2frag_tps import get_grad, get_hess, get_proj, subspace_eig
from mrh.util.la import vector_error
import itertools

def setUpModule():
    global rng
    rng = np.random.default_rng ()

def tearDownModule():
    global rng
    rng = np.random.default_rng ()
    del rng

# TODO: figure out why I have to do x/2, kappa = -kappa/2 here

def seminum (ham_pq, si, lroots, nroots, x):
    proj = get_proj (si, lroots, nroots)
    ene0 = np.dot (si.conj (), np.dot (ham_pq, si))
    grad0 = get_grad (ham_pq, si, lroots, nroots, proj)
    nelu = lroots[1]-nroots
    nelv = lroots[0]-nroots
    kappa_u = np.zeros ((lroots[1],lroots[1]),dtype=x.dtype)
    if nelu:
        kappa_u[nroots:,:nroots] = x[:nelu]
        x = x[nelu:]
    kappa_v = np.zeros ((lroots[0],lroots[0]),dtype=x.dtype)
    if nelv:
        kappa_v[nroots:,:nroots] = x[:nelv]
        x = x[nelv:]
    u = linalg.expm ((kappa_u.T-kappa_u)/2)
    vh = linalg.expm ((kappa_v.T-kappa_v)/2).conj ().T
    ham_pq = ExcitationPSFCISolver.truncrot_ham_pq (None, ham_pq, u, vh)
    x = x/2
    if len (x):
        phi = linalg.norm (x)
        x = np.dot (proj.conj (), x)
        if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
        else: sinp = 1
        si = (np.cos (phi)*si) + (sinp * x)
    grad1 = get_grad (ham_pq, si, lroots, nroots, proj)
    ene1 = np.dot (si.conj (), np.dot (ham_pq, si))
    return grad1-grad0, ene1-ene0

def an (grad, hess, x):
    dg = np.dot (hess, x)
    de = np.dot (grad+.5*dg, x)
    return dg, de

def case (kv, lroots, nroots, q):
    p = np.prod (lroots)
    ham_pq = rng.random (size=(p+q,p+q))
    ham_pq += ham_pq.conj ().T
    si = rng.random (size=(lroots[1],lroots[0]))
    si[nroots:,:] = 0
    si[:,nroots:] = 0
    si = si.ravel ()
    if q: si = np.append (si, rng.random (size=(q)))
    si /= linalg.norm (si)
    proj = get_proj (si, lroots, nroots)
    grad = get_grad (ham_pq, si, lroots, nroots, proj)
    hess = get_hess (ham_pq, si, lroots, nroots, proj)
    x0 = rng.random (size=(grad.size))
    err_tab = np.zeros ((19,4))
    err_str = '\n'
    for ix, p in enumerate (range (19)):
        x1 = x0 / (2**p)
        x1_norm = linalg.norm (x1)
        dg_test, de_test = an (grad, hess, x1)
        dg_ref, de_ref = seminum (ham_pq, si, lroots, nroots, x1)
        de_err = abs ((de_test-de_ref)/de_ref)
        dg_err, dg_theta = vector_error (dg_test, dg_ref)
        err_tab[ix,:] = [x1_norm, de_err, dg_err, dg_theta]
        err_str += '{:e} {:e} {:e} '.format (x1_norm, de_err, dg_err)
        if ix>0:
            de_rel = de_err / err_tab[ix-1,1]
            dg_rel = dg_err / err_tab[ix-1,2]
            err_str += '{:e} {:e}'.format (de_rel, dg_rel)
        err_str += '\n'
        if ix>0 and (abs (de_rel-0.5) < 0.0001) and (abs (dg_rel-0.5) < 0.0001):
            err_tab = err_tab[:ix+1]
            break
    conv_tab = err_tab[1:,:] / err_tab[:-1,:]
    with kv.subTest (q='x'):
        kv.assertAlmostEqual (conv_tab[-1,0], 0.5, 9, msg=err_str)
    with kv.subTest (q='de'):
        kv.assertAlmostEqual (conv_tab[-1,1], 0.5, delta=0.05, msg=err_str)
    with kv.subTest (q='d2e'):
        kv.assertAlmostEqual (conv_tab[-1,2], 0.5, delta=0.05, msg=err_str)



class KnownValues(unittest.TestCase):

    def test_u (self):
        case (self, [2,1], 1, 0)

    def test_v (self):
        case (self, [1,2], 1, 0)

    def test_s (self):
        case (self, [1,1], 1, 1)

    def test_uv (self):
        case (self, [2,2], 1, 0)

    def test_us (self):
        case (self, [2,1], 1, 1)

    def test_vs (self):
        case (self, [1,2], 1, 1)

    def test_full (self):
        case (self, [2,2], 1, 1)

if __name__ == "__main__":
    print("Full Tests for LASSI min_2frag_tps module functions")
    unittest.main()


