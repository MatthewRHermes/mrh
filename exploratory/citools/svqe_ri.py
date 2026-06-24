import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from pyscf.fci.addons import transform_ci_for_orbital_rotation
from pyscf.fci.addons import civec_spinless_repr
from pyscf.fci.direct_spin1 import _unpack_nelec, contract_1e

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
    ''' Get the spinless-basis, density-fitted Hamiltonian '''
    mo = mf.mo_coeff
    nao, nmo = mo.shape
    naux = mf.with_df.get_naoaux ()
    h1_sf = mo.conj ().T @ mf.get_hcore () @ mo
    h1 = np.zeros ((2*nmo, 2*nmo), dtype=h1_sf.dtype)
    h1[:nmo,:nmo] = h1_sf
    h1[nmo:,nmo:] = h1_sf
    h2 = _ao2mo_df (mf.with_df, mo)
    h1 -= lib.einsum ('pik,pkj->ij', h2, h2) / 2
    return h1, h2

def eval_svqe_energy (fcisolver, h1, h2, ci, nelec, fuzzer=None):
    ''' Evaluate the expectation value of spinless-basis, density-fitted Hamiltonian '''
    norb = h1.shape[-1] // 2
    ci = civec_spinless_repr ([ci,], norb, [_unpack_nelec (nelec),])[0]
    nelec = (nelec, 0)
    norb = 2*norb
    if fuzzer is None:
        fuzzer = lambda x: x
    e1, e2 = 0, 0
    ci1 = fuzzer (ci)
    hci1 = contract_1e (h1, ci1, norb, nelec)
    e1 = ci.ravel ().dot (hci1.ravel ())
    for ix, h in enumerate (h2):
        ci1 = fuzzer (fuzzer (fuzzer (fuzzer (fuzzer (ci)))))
        hci1 = contract_1e (h, ci1, norb, nelec)
        e2 += (hci1.ravel ().dot (hci1.ravel ())) / 2
    e = e1 + e2
    return e, e1, e2

def nonunitary_fuzz (r, mag=1e-5, rng=None):
    if rng is None:
        rng = np.random.default_rng ()
    dr = mag * (2*rng.random (r.shape) - 1)
    return r + dr

def unitary_fuzz (r, mag=1e-5, rng=None):
    r = nonunitary_fuzz (r, mag=mag, rng=rng)
    r /= linalg.norm (r)
    return r

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
    h1, h2 = get_svqe_ham (mf)
    ci_hf = np.zeros_like (ci)
    ci_hf[0,0] = 1.0
    rng = np.random.default_rng ()
    e_fci = eval_svqe_energy (fcisolver, h1, h2, ci, nelec)[0] + h0
    e_hf = eval_svqe_energy (fcisolver, h1, h2, ci_hf, nelec)[0] + h0
    print ("sVQE@FCI energy:", e_fci)
    print ("sVQE@HF energy:", e_hf)
    #print ("Using nonunitary fuzz:")
    #def fuzzer (x):
    #    return nonunitary_fuzz (x, mag=1e-3, rng=rng)
    #e_fci_f = eval_svqe_energy (fcisolver, h1, h2, ci, nelec, fuzzer=fuzzer)[0] + h0
    #e_hf_f = eval_svqe_energy (fcisolver, h1, h2, ci_hf, nelec, fuzzer=fuzzer)[0] + h0
    #print ("sVQE@FCI energy:", e_fci_f, e_fci_f - e_fci)
    #print ("sVQE@HF energy:", e_hf_f, e_hf_f - e_hf)
    print ("Using unitary fuzz:")
    def fuzzer (x):
        return unitary_fuzz (x, mag=1e-3, rng=rng)
    e_fci_f = eval_svqe_energy (fcisolver, h1, h2, ci, nelec, fuzzer=fuzzer)[0] + h0
    e_hf_f = eval_svqe_energy (fcisolver, h1, h2, ci_hf, nelec, fuzzer=fuzzer)[0] + h0
    print ("sVQE@FCI energy:", e_fci_f, e_fci_f - e_fci)
    print ("sVQE@HF energy:", e_hf_f, e_hf_f - e_hf)


