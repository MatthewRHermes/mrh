import numpy as np
from pyscf import ao2mo
from pyscf.lib import logger

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
    edm1 += edm1.transpose (0,2,1)

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
    Rorb = sum ([np.dot (v, ed) + np.dot (ev, d) for v, d, ev, ed
        in zip (vj, dm1, evj, edm1)])
    Rorb -= Rorb.T
    
    # CI degree of freedom
    def contract (v,c): return mc.fcisolver.contract_1e (v, c, ncas, nelecas)
    Rci = np.tensordot (const_IJ, ci_arr, axes=1) # Delta_IJ |J> term
    vci = np.stack ([contract (v,c) for v, c in zip (vj, ci)], axis=0)
    for I in range (nroots):
        Rci[I] += 2 * contract (vj[I], Lci[I]) # 2 v_I |J>z_{IJ} term
        Rci[I] += 2 * contract (evj[I], ci[I]) # 2 veff_I |I> term
        Rci[I] -= np.tensordot (L, vci, axes=1) # |W_J>z_{IJ} term
        cc = np.dot (ci[I].ravel ().conj (), Rci[I].ravel ())
        Rci[I] -= ci[I] * cc # Q_I operator

    return mc_grad.pack_uniq_var (Rorb, Rci)

def sarot_grad (mc_grad, Lis, atmlst=atmlst, mo=None, ci=None, eris=None,
        mf_grad=None, **kwargs):
    ''' Returns geometry derivative of Q.x '''

    mc = mc_grad.base
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc = mc_grad.nroots, ncore + ncas
    mo_cas = mo[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = np.tensordot (L, ci_arr, axes=1)

    # Density matrices
    dm1 = np.stack (mc.fcisolver.states_make_rdm1 (ci, ncas, nelecas), axis=0)
    edm1 = np.stack (mc.fcisolver.states_trans_rdm12 (Lci, ci, ncas,
        nelecas)[0], axis=0)
    edm1 += edm1.transpose (0,2,1)
    dm1_ao = reduce (np.dot, (mo_cas, dm1, moH_cas)).transpose (1,0,2)
    edm1_ao = reduce (np.dot, (mo_cas, edm1, moH_cas)).transpose (1,0,2)

    # Potentials and operators
    eri_cas = np.zeros ([ncas,]*4, dtype=dm1.dtype)
    for i in range (ncore, nocc):
        eri_cas[i,:,:,:] = eris.ppaa[i][ncore:nocc,:,:]
    vj = np.tensordot (dm1, eri_cas, axes=2)
    evj = np.tensordot (edm1, eri_cas, axis=2)
    dvj = np.stack (mf_grad.get_jk (mc.mol, list(dm1)), axis=0)
    devj = np.stack (mf_grad.get_jk (mc.mol, list(edm1_ao)), axis=0)

    # Generalized Fock and overlap operator
    gfock = sum ([np.dot (v, ed) + np.dot (ev, d) for v, d, ev, ed
        in zip (vj, dm1, evj, edm1)])
    dme0 = reduce (np.dot, (mo_cas, (gfock+gfock.T)/2, moH_cas))
    s1 = mf_grad.get_ovlp (mc.mol)

    # Crunch
    de_direct = np.zeros ((len (atmlst), 3))
    de_renorm = np.zeros ((len (atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de_renorm[k] -= np.einsum('xpq,pq->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de_direct[k] += np.einsum('xipq,ipq->x', dvj[:,:,p0:p1], edm1_ao[:,p0:p1]) * 2
        de_direct[k] += np.einsum('xipq,ipq->x', devj[:,:,p0:p1], dm1_ao[:,p0:p1]) * 2

    logger.debug (mc, "CMS-PDFT Lis lagrange direct component:\n{}".format (de_direct))
    logger.debug (mc, "CMS-PDFT Lis lagrange renorm component:\n{}".format (de_renorm))
    de = de_direct + de_renorm
    return de


