import numpy as np
from pyscf import ao2mo

def sarot_response (mc_grad, Lis, mo=None, ci=None, eris=None, **kwargs):
    ''' Returns orbital/CI gradient vector '''

    mc = mc_grad.base
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc = mc_grad.nroots, ncore + ncas

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = np.tensordot (L, ci_arr, axes=1)

    # Density matrices
    tril_idx = np.tril_indices (nroots)
    diag_idx = np.arange (nroots)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    tdm1 = np.stack (mc.fcisolver.states_trans_rdm12 (ci_arr[tril_idx[0]],
        ci_arr[tril_idx[1]], ncas, nelecas), axis=0)
    dm1 = tdm1[diag_idx,:,:]
    edm1 = np.stack (mc.fcisolver.states_trans_rdm12 (Lci, ci, ncas,
        nelecas)[0], axis=0)
    edm1 += tdm1.transpose (0,2,1)

    # Potentials
    eri_cas = np.zeros ([ncas,]*4, dtype=dm1.dtype)
    for i in range (ncore, nocc):
        eri_cas[i,:,:,:] = eris.ppaa[i][ncore:nocc,:,:]
    vj = np.tensordot (dm1, eri_cas, axes=2)
    evj = np.tensordot (edm1, eri_cas, axes=2)

    # Constants (state-integrals)
    tvj = np.tensordot (tdm1, eri_cas, axes=2)
    w = np.tensordot (vj, tdm1, axes=((1,2),(1,2)))
    w = ao2mo.restore (1, w, nroots)
    w_IJIJ = np.einsum ('ijij->ij', w)
    w_IIJJ = np.einsum ('iijj->ij', w)
    w_IJJJ = np.einsum ('ijjj->ij', w)
    w_IIII = np.einsum ('iiii->i', w)
    const_IJ = (4*w_IJIJ + 2*w_IIJJ - 2*w_IIII[:,None]) * L
    const_IJ -= np.dot (L, w_IJJJ)

    # Orbital degree of freedom
    Rorb = np.dot (vj, edm1) + np.dot (evj, dm1)
    Rorb -= Rorb.T
    
    # CI degree of freedom
    def contract (v,c): return mc.fcisolver.contract_1e (v, c, ncas, nelecas)
    Rci = np.tensordot (const_IJ, ci_arr, axes=1)
    vci = np.stack ([contract (v,c) for v, c in zip (vj, ci)], axis=0)
    for I in range (nroots):
        Rci[I] += 2 * contract (evj[I], ci[I])
        Rci[I] += 2 * contract (vj[I], Lci[I])
        Rci[I] -= np.tensordot (L, vci, axes=1)
        cc = np.dot (ci[I].ravel ().conj (), Rci[I].ravel ())
        Rci[I] -= ci[I] * cc

    return mc_grad.pack_uniq_var (Rorb, Rci)

def sarot_grad (mc_grad, Lis, atmlst=atmlst, mo=None, ci=None, eris=None,
        mf_grad=None, **kwargs):
    ''' Returns geometry derivative of Q.x '''


