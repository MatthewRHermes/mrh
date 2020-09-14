import sys
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf import fci
from pyscf.fci.direct_spin1 import _unpack_nelec

def addr_outer_product (norb_f, nelec_f):
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    norbrange = np.cumsum (norb_f)
    addrs = cistring.sub_addrs (norb, nelec, range (0,norbrange[0]), nelec_f[0])
    for i in range (1, len (norbrange)):
        addrs = np.union1d (addrs, cistring.sub_addrs (norb, nelec,
            range (norbrange[i-1], norbrange[i]), nelec_f[i]))
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    # There may be an ambiguous factor of -1, but it should apply to the entire product CI vector so maybe it doesn't matter?
    ci_dp = ci_f[-1].copy ()
    for ci_r in ci_f[-2::-1]:
        ndeta_1, ndetb_1 = ci_dp.shape
        ndeta_2, ndetb_2 = ci_r.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r)
        ci_dp = ci_dp.transpose (0,2,1,3).reshape (ndeta_1*ndeta_2, ndetb_1*ndetb_2)
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    ci = np.zeros ((cistring.num_strings (sum (norb_f), sum (neleca_f)), cistring.num_strings (sum (norb_f), sum (nelecb_f))),
        dtype=ci_dp.dtype)
    ci[np.ix_(addrs_a,addrs_b)] = ci_dp[:,:]
    return ci

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    ci_r = []
    for state in range (len (ci_fr[0])):
        ci_f = [ci[state] for ci in ci_fr]
        nelec_f = [nelec[state] for nelec in nelec_fr]
        ci_r.append (_ci_outer_product (ci_f, norb_f, nelec_f))
    nelec = (sum ([ne[0] for ne in nelec_f]),
             sum ([ne[1] for ne in nelec_f]))
    return ci_r, nelec

def slow_ham (mol, h1, h2, ci_fr, norb_f, nelec_fr):
    ci, nelec = ci_outer_product (ci_fr, norb_f, nelec_fr)
    solver = fci.solver (mol)
    norb = sum (norb_f)
    h2eff = solver.absorb_h1e (h1, h2, norb, nelec, 0.5)
    ham_ci = [solver.contract_2e (h2eff, c, norb, nelec) for c in ci]
    ham_eff = np.array ([[c.ravel ().dot (hc.ravel ()) for hc in ham_ci] for c in ci])
    ovlp_eff = np.array ([[bra.ravel ().dot (ket.ravel ()) for ket in ci] for bra in ci])
    return ham_eff, ovlp_eff

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.my_pyscf.mcscf.lasscf_testing import LASSCF
    import os
    class cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    from mrh.examples.lasscf.c2h6n4.c2h6n4_struct import structure as struct
    with cd ("/home/herme068/gits/mrh/examples/lasscf/c2h6n4"):
        mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'sa_lasscf_slow_ham.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    tol = 1e-6 if len (sys.argv) < 2 else float (sys.argv[1])
    las = LASSCF (mf, (4,4), (4,4)).set (conv_tol_grad = tol)
    mo = las.localize_init_guess ((list(range(3)),list(range(9,12))), mo_coeff=mf.mo_coeff)
    las.state_average_(weights = [0.5, 0.5], spins=[[0,0],[2,-2]])
    h2eff_sub, veff = las.kernel (mo)[-2:]
    e_states = las.e_states

    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    mo_coeff = las.mo_coeff
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    e0 = las._scf.energy_nuc () + 2 * (((las._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
    h1 = mo_cas.conj ().T @ (las._scf.get_hcore () + veff.c) @ mo_cas
    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
    nelec_fr = []
    for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
        ne = sum (nelec)
        nelec_fr.append ([_unpack_nelec (fcibox._get_nelec (solver, ne)) for solver in fcibox.fcisolvers])
    ham_eff = slow_ham (las.mol, h1, h2, las.ci, las.ncas_sub, nelec_fr)[0]
    print (las.converged, e_states - (e0 + np.diag (ham_eff)))


