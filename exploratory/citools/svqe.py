import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from pyscf.fci.addons import transform_ci_for_orbital_rotation
from pyscf.fci.addons import civec_spinless_repr
from pyscf.fci.direct_spin1 import _unpack_nelec

def _ao2mo_df_sf (with_df, mo):
    ''' Get the Cholesky vectors in a MO basis. '''
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

def _ao2mo_df (with_df, mo):
    ''' Get the Cholesky vectors in a spinless MO basis. '''
    bPij_sf = _ao2mo_df_sf (with_df, mo)
    naux, nmo = bPij_sf.shape[:2]
    bPij = np.zeros ((naux,2*nmo,2*nmo), dtype=bPij_sf.dtype)
    bPij[:,:nmo,:nmo] = bPij_sf
    bPij[:,nmo:,nmo:] = bPij_sf
    return bPij

def get_svqe_ham (mf):
    ''' From a density-fitted Hartree-Fock object, extract the terms of the Hamiltonian

    H = r1' h1_pp n_p r1 + 1/2 r2I' (h2I_pp n_p)**2 r2I

    where r1 and r2I are orbital rotations and n_p are electron number operators.
    in a spinless basis to avoid ambiguity. In order to match the single-basis
    Hamiltonian operator, h1_pp includes an ``exchange-like'' effect
    from the cumulant decomposition of the 2-body part.
    '''
    mo = mf.mo_coeff
    nao, nmo = mo.shape
    naux = mf.with_df.get_naoaux ()
    h1_sf = mo.conj ().T @ mf.get_hcore () @ mo
    h1 = np.zeros ((2*nmo, 2*nmo), dtype=h1_sf.dtype)
    h1[:nmo,:nmo] = h1_sf
    h1[nmo:,nmo:] = h1_sf
    bPij = _ao2mo_df (mf.with_df, mo)
    r2 = np.empty_like (bPij)
    h2 = np.empty ((naux, 2*nmo), dtype=mo.dtype)
    for p in range (naux):
        h2[p], r2[p] = linalg.eigh (bPij[p])
        h1eff = -h2[p]*h2[p]/2
        h1 += (r2[p] * h1eff[None,:]) @ r2[p].conj ().T
    h1, r1 = linalg.eigh (h1)
    return r1, h1, r2, h2

def eval_svqe_energy (fcisolver, r1, h1, r2, h2, ci, nelec):
    ''' Evaluate the expectation value of the Hamiltonian

    H = r1' h1_pp n_p r1 + 1/2 r2I' (h2I_pp n_p)**2 r2I

    where r1 and r2I are orbital rotations and n_p are electron number operators,
    in a spinless basis to avoid ambiguity.
    '''
    norb = r1.shape[-1] // 2
    ci = civec_spinless_repr ([ci,], norb, [_unpack_nelec (nelec),])[0]
    dm1, dm2 = fcisolver.make_rdm12 (ci, 2*norb, (nelec,0))
    def n1_ci (u):
        return ((dm1 @ u) * u.conj ()).sum (0)
    def n2_ci (u):
        n2 = lib.einsum ('pqrs,sj->pqrj', dm2, u)
        n2 = (n2 * u.conj ()[None,None,:,:]).sum (2)
        n2 = lib.einsum ('pqj,qi->pij', n2, u)
        n2 = (n2 * u.conj ()[:,:,None]).sum (0)
        idx = np.diag_indices_from (n2)
        n2[idx] += n1_ci (u)
        return n2
    e1, e2 = 0, 0
    e1 += np.dot (h1, n1_ci (r1))
    for ix, (u, h) in enumerate (zip (r2, h2)):
        e2 += np.dot (np.dot (n2_ci (u), h), h) / 2
    e = e1 + e2
    return e, e1, e2

def eval_svqe_energy_broken (fcisolver, r1, h1, r2, h2, ci, nelec):
    ''' Evaluate the expectation value of the Hamiltonian

    H = r1' h1_pp n_p r1 + 1/2 r2I' (h2I_pp n_p)**2 r2I

    (in a spinless basis) INCORRECTLY as

    E = r1' h1_pp <n_p> r1 + 1/2 r2I' (h2I_pp <n_p>)**2 r2I

    Explanation: Wick's theorem tells us that

    <n_p n_q> = <n_p> <n_q> - delta_pq <n_p> + <{n_p n_q}>

    The second term is worked into the definition of h1_pp.
    The last term is a correlation correction that only vanishes
        1. for the Hartree--Fock wave function,
        2. in the Hartree--Fock MO basis.
    The expression for E above implicitly omits it.
    '''
    norb = r1.shape[-1] // 2
    ci = civec_spinless_repr ([ci,], norb, [_unpack_nelec (nelec),])[0]
    dm1, dm2 = fcisolver.make_rdm12 (ci, 2*norb, (nelec,0))
    def n1_ci (u):
        return ((dm1 @ u) * u.conj ()).sum (0)
    e1, e2 = 0, 0
    e1 += np.dot (h1, n1_ci (r1))
    for ix, (u, h) in enumerate (zip (r2, h2)):
        n = n1_ci (u)
        e2 += np.dot (h, n)**2 / 2
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
    e_tot += h0
    r1, h1, r2, h2 = get_svqe_ham (mf)
    ci_hf = np.zeros_like (ci)
    ci_hf[0,0] = 1.0
    print ("FCI energy:", e_tot)
    e_fci = eval_svqe_energy (fcisolver, r1, h1, r2, h2, ci, nelec)[0] + h0
    e_hf = eval_svqe_energy (fcisolver, r1, h1, r2, h2, ci_hf, nelec)[0] + h0
    print ("sVQE@FCI energy:", e_fci)
    print ("sVQE@HF energy:", e_hf)
    print ("Using the incorrect energy expression:")
    e_fci = eval_svqe_energy_broken (fcisolver, r1, h1, r2, h2, ci, nelec)[0] + h0
    e_hf = eval_svqe_energy_broken (fcisolver, r1, h1, r2, h2, ci_hf, nelec)[0] + h0
    print ("sVQE@FCI energy:", e_fci)
    print ("sVQE@HF energy:", e_hf)


