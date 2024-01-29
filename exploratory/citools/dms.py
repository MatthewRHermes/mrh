import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.direct_spin1 import make_rdm12s
from mrh.my_pyscf.mcpdft._dms import dm2_cumulant
from mrh.my_pyscf.fci.csfstring import CSFTransformer

def random_ci_vector (norb, nelec, csf=False, csf_off=0):
    neleca, nelecb = _unpack_nelec (nelec)
    na = cistring.num_strings (norb, neleca)
    nb = cistring.num_strings (norb, nelecb)
    ci0 = np.random.rand (na,nb)
    ci0 = ci0 / linalg.norm (ci0)
    if csf:
        try:
            smult = (neleca - nelecb) + 1 + 2*csf_off
            t = CSFTransformer (norb, neleca, nelecb, smult)
            ci0 = t.vec_det2csf (ci0)
            ci0 = t.vec_csf2det (ci0)
        except AssertionError as e:
            return np.zeros ((na,nb))
    return ci0

def random_1det_rdms (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    dm1s = []
    for n in (neleca, nelecb):
        kappa = np.random.rand (norb, norb)
        kappa -= kappa.T
        umat = linalg.expm (kappa) [:,:n]
        dm1s.append (umat @ umat.T)
    assert (abs (np.trace (dm1s[0])-neleca) < 1e-8), dm1s[0]
    assert (abs (np.trace (dm1s[1])-nelecb) < 1e-8), dm1s[1]
    dm1 = dm1s[0] + dm1s[1]
    dm2 = np.multiply.outer (dm1, dm1)
    for d in dm1s:
        dm2 -= np.multiply.outer (d, d).transpose (0,3,2,1)
    return dm1s, dm2

def random_rohf_rdms (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    kappa = np.random.rand (norb, norb)
    kappa -= kappa.T
    umat = linalg.expm (kappa) 
    umata, umatb = umat[:,:neleca], umat[:,:nelecb]
    dm1s = [umata @ umata.T, umatb @ umatb.T]
    assert (abs (np.trace (dm1s[0])-neleca) < 1e-8), dm1s[0]
    assert (abs (np.trace (dm1s[1])-nelecb) < 1e-8), dm1s[1]
    dm1 = dm1s[0] + dm1s[1]
    dm2 = np.multiply.outer (dm1, dm1)
    for d in dm1s:
        dm2 -= np.multiply.outer (d, d).transpose (0,3,2,1)
    return dm1s, dm2

def daniels_fn (dm1s, dm2, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    dm1 = dm1s[0] + dm1s[1]
    dm1n = (2-(neleca+nelecb)/2.) * dm1 - np.einsum('pkkq->pq', dm2)
    dm1n *= 2./(neleca-nelecb+2)
    return dm1n

if __name__=='__main__':
    for norb in range (1,7):
        for neleca in range (1,norb+1):
            for nelecb in range (neleca+1):
                #if norb > 2: break
                if neleca==norb and nelecb==norb: continue
                ci0 = random_ci_vector (norb, (neleca, nelecb), csf=True)
                if linalg.norm (ci0) == 0:
                    continue
                dm1s, dm2s = make_rdm12s (ci0, norb, (neleca, nelecb))
                dm2 = dm2s[0] + dm2s[1] + dm2s[2] + dm2s[1].transpose (2,3,0,1)
                sm1_ref = dm1s[0] - dm1s[1]
                sm1_test = daniels_fn (dm1s, dm2, (neleca, nelecb))
                try:
                    sm1_err = linalg.norm (sm1_test-sm1_ref)
                    if np.abs (sm1_err) < 1e-8: sm1_err = "yes"
                except ValueError as e:
                    sm1_err = str (e)
                is_1det = (nelecb == 0) or (neleca == norb)
                print ("general", norb, (neleca,nelecb), is_1det, sm1_err)
                #dm1 = dm1s[0] + dm1s[1]
                #cm2 = dm2_cumulant (dm2, dm1s)
                #print ("2D_pq =", 2*dm1)
                #print ("l_prrq =", np.einsum ('prrq->pq', cm2))
                #print ("D_pr D_qr =", dm1 @ dm1.T)
                #print ("2D_pq - l_prrq - D_pr D_qr =", 2*dm1 - dm1 @ dm1.T - np.einsum ('prrq->pq', cm2))
                #print ("M_pq test =", sm1_test)
                #print ("M_pq ref =", sm1_ref)
                dm1s, dm2 = random_1det_rdms (norb, (neleca, nelecb))
                sm1_ref = dm1s[0] - dm1s[1]
                sm1_test = daniels_fn (dm1s, dm2, (neleca, nelecb))
                try:
                    sm1_err = linalg.norm (sm1_test-sm1_ref)
                    if np.abs (sm1_err) < 1e-8: sm1_err = "yes"
                except ValueError as e:
                    sm1_err = str (e)
                #print ("1det", norb, (neleca,nelecb), sm1_err)
                dm1s, dm2 = random_rohf_rdms (norb, (neleca, nelecb))
                sm1_ref = dm1s[0] - dm1s[1]
                sm1_test = daniels_fn (dm1s, dm2, (neleca, nelecb))
                try:
                    sm1_err = linalg.norm (sm1_test-sm1_ref)
                    if np.abs (sm1_err) < 1e-8: sm1_err = "yes"
                except ValueError as e:
                    sm1_err = str (e)
                #print ("ROHF", norb, (neleca,nelecb), sm1_err)


