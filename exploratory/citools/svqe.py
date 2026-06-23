import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from pyscf.fci.addons import transform_ci_for_orbital_rotation

def _ao2mo_df (with_df, mo):
    naux = with_df.get_naoaux ()
    nao, nmo = mo.shape
    bPij = np.empty ((naux, nmo, nmo), dtype=mo.dtype)
    ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (mo, mo, compact=False)
    b0 = 0
    for eri1 in with_df.loop ():
        b1 = b0 + eri1.shape[0]
        eri2 = bPij[b0:b1]
        eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
        b0 = b1
    #h2_test = lib.einsum ('Pij,Pkl->ijkl',bPij,bPij)
    #h2_ref = ao2mo.restore (1, with_df.ao2mo (mo), nmo)
    #assert (np.amax (np.abs (h2_test-h2_ref)) < 1e-8)
    return bPij

def get_svqe_ham (mf):
    mo = mf.mo_coeff
    nao, nmo = mo.shape
    naux = mf.with_df.get_naoaux ()
    h1 = mo.conj ().T @ mf.get_hcore () @ mo
    bPij = _ao2mo_df (mf.with_df, mo)
    u2 = np.empty_like (bPij)
    h2 = np.empty ((naux, nmo), dtype=mo.dtype)
    for p in range (naux):
        h2[p], u2[p] = linalg.eigh (bPij[p])
        h1eff = -h2[p]*h2[p]/2
        h1 += (u2[p] * h1eff[None,:]) @ u2[p].conj ().T
    h1, u1 = linalg.eigh (h1)
    return u1, h1, u2, h2

def eval_svqe_energy (fcisolver, u1, h1, u2, h2, ci, nelec):
    norb = u1.shape[-1]
    def n_ci (u):
        ci1 = transform_ci_for_orbital_rotation (ci.copy (), norb, nelec, u)
        return fcisolver.make_rdm1 (ci1, norb, nelec).diagonal ()
    e1, e2 = 0, 0
    e1 += np.dot (h1, n_ci (u1))
    for u, h in zip (u2, h2):
        e2 += (np.dot (h, n_ci (u))**2) / 2
    e = e1 + e2
    return e, e1, e2

if __name__=='__main__':
    from pyscf import gto, scf, fci
    xyz = '''O 0.0000000 0.0000000 -0.3893611
             H 0.7629844 0.0000000 0.1946806
             H -0.7629844 0.0000000 0.1946806'''
    mol = gto.M (atom=xyz, basis='sto-3g')
    mf = scf.RHF (mol).density_fit ().run ()
    nelec = mol.nelectron
    norb = mol.nao_nr ()
    fcisolver = fci.solver (mol)
    mo = mf.mo_coeff
    h0 = mf.energy_nuc ()
    h1 = mo.conj ().T @ mf.get_hcore () @ mo
    h2 = mf.with_df.ao2mo (mo)
    e_tot, ci = fcisolver.kernel (h1, h2, norb, nelec)
    u1, h1, u2, h2 = get_svqe_ham (mf)
    e_test, e1, e2 = eval_svqe_energy (fcisolver, u1, h1, u2, h2, ci, nelec)
    print ("FCI energy:", e_tot + h0)
    print ("sVQE energy:", e_test + h0)

