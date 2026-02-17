import os
import sys
import numpy as np
from functools import reduce

from pyscf import lib, __config__

from pyscf.pbc.lib import kpts_helper

'''
I think these reference are utmost useful, if anyone is trying to implement the CASSCF.
1.
2.
3.
4.
5.
6.
'''

'''
Steps
1. Generalize the gen_g_hop, heassian_op and hessian_diag functions to the k-point case.
2. Integrate the above functions with k-point CIAH solver.
'''

def gen_g_hop(mc, mo_coeff, mo_phase, u, casdm1, casdm2, eris):
    '''
    To solve the second order or quasi-second order CASSCF equations, we need to generate the gradient and the Hessian-vector product. I am generalizing pyscf/mcscf/mc1step.py to the k-point case.
    Note that the input args are different than the original gen_g_hop function.
    '''
    ncas = mc.ncas
    nelecas = mc.nelecas
    nkpts = mc.nkpts
    ncore = mc.ncore
    nocc = ncore + ncas
    nmo = mo_coeff[0].shape[1]
    dtype = casdm1.dtype
    ncasncas = ncas*ncas
    nmonmo = nmo*nmo

    kconserv = kpts_helper.get_kconserv(mc._scf.cell, mc._scf.kpts)

    assert casdm1.shape == (nkpts*ncas, nkpts*ncas)
    assert casdm2.shape == (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
    assert casdm1.dtype == casdm2.dtype

    # This is in MO basis
    dm1 = np.empty((nkpts, nmo, nmo), dtype=casdm1.dtype)
    CASDM1_k = np.empty((nkpts, ncas, ncas), dtype=casdm1.dtype)
    idx = np.arange(ncore)
    
    for k in range(nkpts):
        dm1[k][idx, idx] = 2.0
        casdm1_k = reduce(np.dot, (mo_phase[k], casdm1, mo_phase[k].conj().T))
        dm1[k][ncore:nocc, ncore:nocc] = casdm1_k
        CASDM1_k[k] = casdm1_k
    
    CASDM2_k = np.empty((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=casdm1.dtype)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv(k1, k2, k3)
        dm2_k = np.einsum('ip, jq, pqrs, kr, ls->ijkl', mo_phase[k1], mo_phase[k2], casdm2, mo_phase[k3], mo_phase[k4])
        CASDM2_k[k1, k2, k3] = dm2_k

    jkcaa = np.zeros((nkpts, nocc, ncas), dtype=dtype)
    vhf_a = np.zeros((nkpts, nmo, nmo), dtype=dtype)
    g_dm2 = np.zeros((nkpts, nmo, ncas), dtype=dtype)

    hdm2 = np.empty((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    
    sl = np.arange(nocc)
    reshape_ = (nmo, nmo, ncas, ncas)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]

        # I think we should not loop over nmo:
        # First I made the without for loop version for the molecular case.
        # vhf_a = np.einsum('iquv,uv->iq', ppaa, casdm1) 
        # vhf_a -= 0.5*np.einsum('iuqv,uv->iq', papa, casdm1)
        # slice = np.arange(nocc)
        # temp = (6.0*papa[:nocc, :, :nocc, :] - 2.0*ppaa[:nocc, :nocc, :, :])[slice, :, slice, :]
        # jkcaa = np.einsum('iuv,uv->iu', temp, casdm1)
        # casdm2_mat = casdm2.reshape(ncas*ncas, ncas*ncas)
        # _reshape = (nmo, nmo, ncas, ncas)
        # jtmp_all = lib.dot(ppaa.reshape(nmo*nmo, ncas*ncas), casdm2_mat).reshape(_reshape)
        # ktmp_all = lib.dot(papa.transpose(0,2,1,3).reshape(nmo*nmo, ncas*ncas), dm2tmp).reshape(_reshape)
        # hdm2 = (ktmp_all + jtmp_all).transpose(0, 2, 1, 3)
        # g_dm2 = np.einsum('iaav->iv', jtmp_all[:, ncore:nocc, :, :])

        jbuf = eris.ppaa[k1, k2, k3]
        kbuf = eris.papa[k1, k2, k3]
        dm1 = CASDM1_k[k4]
        dm2 = CASDM2_k[k1, k2, k3]

        dm2t = (dm2.conj().transpose(1,2,0,3) + dm2.conj().transpose(0,2,1,3)).reshape(ncasncas, ncasncas)
        vhf_a[k4] += (np.einsum('iquv,uv->iq', jbuf, dm1)
                    - 0.5*np.einsum('iuqv,uv->iq', kbuf, dm1))

        temp = (6.0*kbuf[:nocc, :, :nocc, :] - 2.0*jbuf[:nocc, :nocc, :, :])[sl, :, sl, :]
        jkcaa[k4] += np.einsum('iuv,uv->iu', temp, dm1)

        dm2m = dm2.reshape(ncasncas, ncasncas)
        jtmp = lib.dot(jbuf.reshape(nmonmo, ncasncas), dm2m).reshape(reshape_)
        g_dm2[k4] += np.einsum('iuuv->iv', jtmp[:, ncore:nocc, :, :])
        
        ktmp = lib.dot(kbuf.transpose(0,2,1,3).reshape(nmonmo, ncasncas), dm2t).reshape(reshape_)
        hdm2[k1, k2, k3] = (ktmp + jtmp).transpose(0, 2, 1, 3)
    
    jbuf = kbuf = jtmp = ktmp = temp = None
    
    vhf_ca = np.empty_like(vhf_a)
    h1e_mo = np.empty((nkpts, nmo, nmo), dtype=dtype)
    g = np.zeros_like(h1e_mo, dtype=dtype)

    hcore = mc.get_hcore()
    for k in range(nkpts):
        vhf_ca[k] = eris.vhf_c[k] + vhf_a[k]
        h1e_mo[k] = reduce(np.dot, (mo_coeff[k].conj().T, hcore[k], mo_coeff[k]))
        g[k][:,:ncore] = 2.0 * (h1e_mo[k][:,:ncore] + vhf_ca[k][:,:ncore])
        g[k][:,ncore:nocc] = np.dot(h1e_mo[k][:, ncore:nocc] + eris.vhf_c[k][:, ncore:nocc], CASDM1_k[k])
        g[k][:,ncore:nocc] += g_dm2[k]
    
    hcore = None

    def gorb_update(u, fcivec):
        pass
    
    # Hessian diagonal
    hdiag = np.empty((nkpts, nmo, ncas), dtype=dtype)
    
    for k in range(nkpts):
        temp = np.einsum('ii, jj->ij', h1e_mo[k], CASDM1_k[k])
        temp -= h1e_mo[k] * CASDM1_k[k]
        hdiag[k] = temp + temp.conj().T
        g_diag = g[k].diagonal()
        hdiag[k] -= g_diag + g_diag.reshape(-1, 1)
        idx = np.arange(nmo)
        hdiag[k][idx, idx] += 2.0 * g_diag 
        v_diag = vhf_ca[k].diagonal()
        hdiag[k][:, :ncore] += 2.0 * v_diag.reshape(-1, 1)
        hdiag[k][:ncore] += 2.0 * v_diag
        idx = np.arange(ncore)
        hdiag[k][idx, idx] -= 4.0 * v_diag[:ncore]

        tmp = np.einsum('ii,jj->ij', eris.vhf_c[k], CASDM1_k[k])
        hdiag[k][:, ncore:nocc] += tmp
        hdiag[k][ncore:nocc, :] += tmp.conj().T
        tmp = -eris.vhf_c[k][ncore:nocc,ncore:nocc] * CASDM1_k[k]
        hdiag[k][ncore:nocc,ncore:nocc] += tmp + tmp.conj().T
        tmp = 6 * eris.k_pc[k] - 2 * eris.j_pc[k]
        hdiag[k][ncore:,:ncore] += tmp[ncore:]
        hdiag[k][:ncore,ncore:] += tmp[ncore:].conj().T
        hdiag[k][:nocc,ncore:nocc] -= jkcaa[k]
        hdiag[k][ncore:nocc,:nocc] -= jkcaa[k].conj().T

        v_diag = np.einsum('ijij->ij', hdm2[k])
        hdiag[k][ncore:nocc,:] += v_diag.conj().T
        hdiag[k][:,ncore:nocc] += v_diag

    g_orb = np.hstack([mc.pack_uniq_var(g[k], g[k].conj().T) 
                       for k in range(nkpts)])
    h_diag = np.hstack([mc.pack_uniq_var(hdiag[k]) 
                        for k in range(nkpts)])
    
    def h_op(x):
        x2=None
        return mc.pack_uniq_var(x2)
    
    return g_orb, gorb_update, h_op, h_diag