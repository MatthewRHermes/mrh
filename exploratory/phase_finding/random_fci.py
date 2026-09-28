import numpy as np
from scipy.sparse import linalg as sparse_linalg
from pyscf import gto, scf, fci, ao2mo
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.csf_fci import csf_solver

rng = np.random.default_rng ()

def random_ham (norb):
    h1 = rng.random ([norb,]*2)
    h2 = rng.random ([norb,]*4)
    h1 += h1.T
    h2 += h2.transpose (2,3,0,1)
    h2 += h2.transpose (1,0,3,2)
    return h1, ao2mo.restore (8, h2, norb)

# We do this to make sure we are using canonical orbitals, even with a random Hamiltonian
def random_mol_rhf (norb, nelec):
    mol = gto.Mole ()
    mol.verbose = 0
    mol.stdout = '/dev/null'
    mol.atom.append (('C', (0, 0, 0)))
    mol.nelectron = nelec
    mol.incore_anyway = True
    mol.nao = lambda * args : norb
    mol.nao_nr = lambda * args : norb
    mol.build ()
    mf = scf.RHF (mol)
    h1, h2 = random_ham (norb)
    mf.get_hcore = lambda * args : h1
    mf.get_ovlp = lambda * args : np.eye (norb)
    mf.energy_nuc = lambda * args : 0
    mf._eri = h2
    mf.init_guess = '1e'
    mf.kernel ()
    return mf

def get (norb, nelec):
    for i in range (10):
        myerr = None
        try:
            mf = random_mol_rhf (norb, nelec)
            myerr = None
        except np.linalg.LinAlgError as err:
            myerr = err
        if myerr is None:
            break
    if myerr is not None:
        raise (myerr)
    myfci = csf_solver (mf.mol, smult=1)
    h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
    h2 = ao2mo.full (mf._eri, mf.mo_coeff)
    e, c = myfci.kernel (h1, h2, norb, nelec, nroots=10)
    t = myfci.transformer
    c = np.asarray (t.vec_det2csf (c)).reshape (-1, t.ncsf).T
    h_diag = myfci.make_hdiag_csf (h1, h2, norb, nelec)
    h2eff = myfci.absorb_h1e (h1, h2, norb, nelec)
    linkstrl = myfci.gen_linkstr (norb, nelec, True)
    def _h_op (x):
        x = t.vec_csf2det (x, normalize=False)
        hx = myfci.contract_2e (h2eff, x, norb, nelec, link_index=linkstrl)
        hx = t.vec_det2csf (hx, normalize=False)
        return hx
    h_op = sparse_linalg.LinearOperator (
        matvec=_h_op,
        shape=(t.ncsf,t.ncsf),
        dtype=c.dtype
    )
    return e, c, h_op, h_diag

