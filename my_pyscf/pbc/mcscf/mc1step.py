
import sys
import numpy as np
import scipy
from functools import reduce

from pyscf import lib, __config__
from pyscf.soscf import ciah # Recently they have added the CIAH solver for PBC. Will use it!
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.mcscf.mc1step import CASSCF as molCASSCF
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf.mc_ao2mo import _ERIS
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R
from mrh.my_pyscf.pbc.mcscf import casci as casciModule

logger = lib.logger

# Author: Bhavnesh Jangid

'''
I think these reference are utmost useful, if anyone is trying to implement the CASSCF.
1. Chem Phy. 1980, 48, 157-173
2. Phys. Scripta, 1980, 21, 323-327
3. Theo Chem Acc 1997, 97, 88-95
4. CPL 2017, 683, 291-299
5. JCP 2019, 150, 194106
6. JCP 2019, 152, 074102
7. IJQC, 2009, 109, 2178-2190 (For DIIS)
8. JCC. 2018, 40, 1463-1470 (For DIIS)
'''

'''
Steps
1. Generalize the gen_g_hop, heassian_op and hessian_diag functions to the k-point case.
2. Integrate the above functions with k-point CIAH solver.
'''

'''
#TODOs:
1. Normalize the 2e integrals in mc_ao2mo generation only.
2. 
'''

def _get_casdm2_kpts(casdm2, mo_phase1, klabel):
    '''
    Compute the 2RDM for a given k-point configuration.
    '''
    k1, k2, k3, k4 = klabel
    dm2_k = np.einsum('iP, jQ, PQRS, kR, lS->ijkl', 
                        mo_phase1[k1].conj(), mo_phase1[k2], 
                        casdm2, 
                        mo_phase1[k3].conj(), mo_phase1[k4])
    return dm2_k

def gen_g_hop(mc, mo_coeff, mo_phase, u, casdm1, casdm2, eris):
    '''
    To solve the second order or quasi-second order CASSCF equations, we need to 
    generate the gradient, hessian diagonal and the Hessian-vector product. 
    I am generalizing pyscf/mcscf/mc1step.py to the k-point case. 
    Note that the input args are different than the original gen_g_hop function.
    PySCF implementation doesn't have the docstring, but I am writing one to make my or
    future developer life easier.
    args:
        mc: casscf object
            instance of the CASSCF class.
        mo_coeff: list of np.ndarray (nkpts, nao, nmo) (block orbitals)
            List of the MO coefficients for each k-point.
        mo_phase: list of np.ndarray (nkpts, ncas, ncastot)
            List of the phase factors for transforming the block orbitals to wannier orbitals.
            This is used to transform the casdm1 and casdm2 to block orbital basis, which is then used
            to compute the grad, hessian-vector product and hessian diagonal.
            Look below for the description of the various variables.
        u: list of np.ndarray (nkpts, nmo, nmo) (block orbitals)
            orbital rotation matrix for each k-point.
        casdm1: np.ndarray (ncastot, ncastot) (wannier orbitals)
            1-RDM in the CAS space. This is in the wannier MO basis.
        casdm2: np.ndarray (ncastot, ncastot, ncastot, ncastot) (wannier orbitals)
            2-RDM in the CAS space. This is in the wannier MO basis.
        eris: ao2mo object for the casscf object.
            Saved Attributes in the eris object are:
            All of them in the block orbital basis.
            ppaa: np.array (nkpts, nkpts, nkpts, nmo, nmo, ncas, ncas)/ read from disk 
                It's a function that takes k1, k2, k3 as input and returns the ppaa integrals
            papa: np.array (nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas)/ read from disk
                It's a function that takes k1, k2, k3 as input and returns the papa integrals
            j_pc: np.array (nkpts, nmo, ncore)
                potential due to core electrons
            k_pc: np.array (nkpts, nmo, ncore)
                potential due to core electrons
            vhf_c: np.array (nkpts, nmo, nmo)
                VHF matrix due to core electrons
        
        Additionally, some other variables used in this functions are:
        Note, ncore, ncas, nelecas is for the unit-cell not for the whole supercell.
            ncas: int
                Number of active space orbitals.
            nelecas: tuple of int
                Number of active space electrons. (nalpha, nbeta)
            nkpts: int
                Number of k-points.
            ncastot: int
                Total number of active space orbitals in the supercell. 
                This is equal to ncas * nkpts.
            ncore: int
                Number of core orbitals.
    returns:
        Note: The output shapes of these objects are converted to 1D arrays to be compatible with
        Gamma-CIAH solver. However, once debugged I will use the k-CIAH, in that case the shapes would be as mentioned below.
        g_orb: list of np.ndarray (nkpts, nmo, nmo)
            Orbital gradient for each k-point. (block orbital basis)
        gorb_update: function 
            Function to update the orbital gradient after the orbital rotation.
        h_op: function
            Function to compute the Hessian-vector product. (block orbital basis)
        h_diag: np.ndarray (nkpts, nmo, ncas) (block orbital basis)
            Diagonal of the Hessian matrix. This is used for preconditioning in the Davidson solver.
    '''

    kmf = mc._scf
    cell = kmf.cell
    ncas = mc.ncas
    nelecas = mc.nelecas
    nkpts = mc.nkpts
    ncore = mc.ncore
    nocc = ncore + ncas
    nmo = mo_coeff[0].shape[1]
    dtype = casdm1.dtype

    kpts = kmf.kpts
    
    ncastot = nkpts*ncas

    kconserv = kpts_helper.get_kconserv(cell, kpts)

    log = logger.new_logger(mc, mc.verbose)

    if log.verbose >= logger.DEBUG:
        assert casdm1.shape == (ncastot, ) * 2
        assert casdm2.shape == (ncastot, ) * 4
        assert casdm1.dtype == casdm2.dtype

    # First convert the casdm1 and casdm2 to k-space.
    # The constructed dm1 would be in block orbitals (MO basis).
    dm1 = np.zeros((nkpts, nmo, nmo), dtype=casdm1.dtype)
    casdm1_kpts = np.zeros((nkpts, ncas, ncas), dtype=dtype)
    idx = np.arange(ncore)
    
    for k in range(nkpts):
        dm1[k][idx, idx] = 2.0
        casdm1_k = reduce(np.dot, (mo_phase[k], casdm1, mo_phase[k].conj().T))
        dm1[k][ncore:nocc, ncore:nocc] = casdm1_k
        casdm1_kpts[k] = casdm1_k
    
    casdm2_kpts = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=dtype)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        dm2_k = _get_casdm2_kpts(casdm2, mo_phase, (k1, k2, k3, k4))
        casdm2_kpts[k1, k2, k3] = dm2_k

    # Sanity checks.
    if log.verbose >= logger.DEBUG1:
        log.debug("Number of electrons in the CAS space: %s",  
              sum([casdm1_kpts[k].trace() for k in range(nkpts)]).real)
        for k in range(nkpts):
            log.debug1("Number of electrons in the CAS space for k-point %s: %s", 
                       k, casdm1_kpts[k].trace().real)
        
        log.debug("Number of electron: %s", np.einsum('ppqq->', casdm2).real)
        log.debug("Number of electrons in the CAS space from casdm2: %s", 
                   sum([np.einsum('ppqq->', casdm2_kpts[k1,k1,k3]) 
                        for k1 in range(nkpts) 
                        for k3 in range(nkpts)]).real)
    
    # Step-1:Orbital Gradients
    # Construct the potential
    vhf_a = np.zeros((nkpts, nmo, nmo), dtype=dtype)
    g_dm2 = np.zeros((nkpts, nmo, ncas), dtype=dtype)

    # Collect the contribution from different k1, k2 and k3 whenever they are
    # equals to momentum of the output entity.
    # I think we should not loop over nmo, because it will be solved for 
    # a given k-point, means the number of orbitals would be way small than total system. 
    # To remove the loop over nmo, as done in the molecular code, I have first converted 
    # that (molecular) code without for loop, matched it with loop.
    # That code is:
    # ppaa = eris.ppaa
    # papa = eris.papa
    # vhf_a = numpy.einsum('pquv,uv->pq', ppaa, casdm1)
    # vhf_a -= 0.5*numpy.einsum('puqv,uv->pq', papa, casdm1)
    # jtmp = lib.dot(ppaa.reshape(nmo*nmo,-1), casdm2.reshape(ncas*ncas,-1))
    # jtmp = jtmp.reshape(nmo,nmo,ncas,ncas) # nmo, nmo, ncas, ncas
    # g_dm2 = numpy.einsum('puuv->pv', jtmp[:, ncore:nocc])

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        ppaa = eris.ppaa(k1, k2, k3) # (k1, k2, k3, k4)

        if (k1 == k2) and (k3==k4):
            vhf_a[k1] +=  1.0/nkpts * np.einsum('pquv,vu->pq', ppaa, casdm1_kpts[k3]) # (k1,k1)

        if (k1 == k4) and (k2 == k3):
            paap = eris.paap(k1, k2, k3) # (k1, k2, k3, k4)
            vhf_a[k1] -= 0.5/nkpts * np.einsum('puvq,uv->pq', paap, casdm1_kpts[k2]) # (k1,k1)
            
        dm2_blk = casdm2_kpts[k1, k2, k3]   # (k1, k2, k3, k4)
        jtmp = 1/nkpts * np.einsum('pqvw,tuvw->pqut', ppaa[:, ncore:nocc, :, :], dm2_blk)
        g_dm2[k1] += np.einsum('puuv->pv', jtmp)

    ppaa = dm2_blk = paap = jtmp = None
    
    # Now assemble the pieces and construct the gradient.    
    hcore = mc.get_hcore() # (nkpts, nao, nao)
    
    vhf_ca = np.array([eris.vhf_c[k] + vhf_a[k] 
                       for k in range(nkpts)], dtype=dtype) # (nkpts, nmo, nmo) (block orbital basis)
    h1e_mo = np.array([reduce(np.dot, (mo_coeff[k].conj().T, hcore[k], mo_coeff[k])) 
                       for k in range(nkpts)], dtype=dtype) # (nkpts, nmo, nmo) (block orbital MO basis)
    hcore = None

    g = np.zeros((nkpts,nmo,nmo), dtype=dtype)
    for k in range(nkpts):
        g[k][:,:ncore] = 2.0 * (h1e_mo[k][:,:ncore] + vhf_ca[k][:,:ncore])
        g[k][:,ncore:nocc] = np.dot(h1e_mo[k][:, ncore:nocc] + eris.vhf_c[k][:, ncore:nocc], casdm1_kpts[k])
        g[k][:,ncore:nocc] += g_dm2[k]

    # Step-2: Gradient update function
    # In second order one-step orbital optimization, in the micro iterations : the gradient is updated
    # with the fixed hessians and rotated set of orbitals rather than transforming the 2e integrals again.

    def gorb_update(u, fcivec):
        # TODO: Note: currently I am using the CIAH not the k-CIAH, so the update matrix is packed into
        # one giant matrix. This will need restructured once I switch to k-CIAH.

        u = block_diag_to_kblocks(u, nkpts, nmo)
        assert u.shape == (nkpts, nmo, nmo)

        mo1 = np.array([np.dot(mo_coeff[k], u[k]) 
                        for k in range(nkpts)], dtype=dtype) # (nkpts, nao, nmo)
        mo_phase1 = get_mo_coeff_k2R(kmf, mo1, ncore, ncas)[-1]

        # Compute the RDMs        
        ncastot = nkpts * ncas
        nelectot = (nkpts*nelecas[0], nkpts*nelecas[1])
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncastot, nelectot) # Wannier basis
                
        # To update the gradients:
        nao = mo_coeff[0].shape[0]
        g = np.zeros((nkpts, nmo, nmo), dtype=dtype)
        dm_core_kpts = np.empty((nkpts, nao, nao), dtype=dtype)
        dm_act_kpts =  np.empty((nkpts, nao, nao), dtype=dtype)

        for k in range(nkpts):
            uc = u[k][:, :ncore].copy()
            ua = u[k][:, ncore:nocc].copy()
            mo_c = np.dot(mo_coeff[k], uc)
            mo_a = np.dot(mo_coeff[k], ua)
            dm_core_kpts[k] = 2.0 * np.dot(mo_c, mo_c.conj().T)
            casdm1_k = reduce(np.dot, (mo_phase1[k], casdm1, mo_phase1[k].conj().T))
            dm_act_kpts[k] = reduce(np.dot, (mo_a, casdm1_k, mo_a.conj().T))

        # Now compute the vj and vk for the core and active density matrices separately, 
        # then contract with the mo1 to get the vhf in the mo1 basis.
        vj_k, vk_k = mc._scf.get_jk(cell, dm_kpts=dm_core_kpts, hermi=1, 
                                    with_j=True, with_k=True, kpts=kpts, exxdiv=None)

        vhf_c = np.array([reduce(
            np.dot, (mo1[k].conj().T, vj_k[k]-vk_k[k]*0.5, mo1[k][:,:nocc])) 
            for k in range(nkpts)], dtype=dtype)
        
        vj_k, vk_k = mc._scf.get_jk(cell, dm_kpts=dm_act_kpts, hermi=1, 
                                    kpts=kpts, with_j=True, with_k=True,exxdiv=None)
        vhf_a = np.array([reduce(
            np.dot, (mo1[k].conj().T, vj_k[k]-vk_k[k]*0.5, mo1[k][:,:nocc])) 
            for k in range(nkpts)], dtype=dtype)

        for k in range(nkpts):
            casdm1_k = reduce(np.dot, (mo_phase1[k], casdm1, mo_phase1[k].conj().T))
            h1e_mo1k = reduce(np.dot, (u[k].conj().T, h1e_mo[k], u[k][:,:nocc]))
            g[k][:, :ncore] = 2.0 * (h1e_mo1k[:,:ncore] + vhf_c[k][:,:ncore] + vhf_a[k][:,:ncore])
            g[k][:,ncore:nocc] = np.dot(h1e_mo1k[:,ncore:nocc] + vhf_c[k][:,ncore:nocc], casdm1_k)
        
        vj_k = vk_k = vhf_a = vhf_c = h1e_mo1k = dm_core_kpts = dm_act_kpts = None

        # 2e part of the gradient update.
        # These objects would be insanly huge. Instead of creating them and storing, I think I should 
        # compute them on the fly whenever they are required.
        # Benchmarked on the molecular code.
        # ppaa = eris.ppaa
        # papa = eris.papa
        # p1aa = numpy.einsum('pr, tq, rquv-> ptuv', u.T, ua.T, ppaa)
        # paa1 = numpy.einsum('pr, ruvq, qt -> puvt', u.T, papa.transpose(0,1,3,2), ra)
        # p1aa += paa1
        # p1aa += paa1.transpose(0,1,3,2)
        # g[:, :ncore:nocc] += np.einsum('puwx, wxuv->pu', p1aa, casdm2)
        for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
            k4 = kconserv[k1, k2, k3]
            ppaa = eris.ppaa(k1, k2, k3)
            papa = eris.papa(k1, k2, k3)
            paap = eris.paap(k1, k2, k3)
            ua  = u[k2][:, ncore:nocc]
            ra3 = (u[k3] - np.eye(nmo, dtype=dtype))[:, ncore:nocc]
            ra4 = (u[k4] - np.eye(nmo, dtype=dtype))[:, ncore:nocc]
            p1aa = 1/nkpts * np.einsum('pr,tq,rquv->ptuv', u[k1].conj().T, ua.T, ppaa, optimize=True)
            pa1a = 1/nkpts * np.einsum('pr,ruqv,qt->putv', u[k1].conj().T, papa, ra3.conj(), optimize=True)
            paa1 = 1/nkpts * np.einsum('pr,rvuq,qt->pvtu', u[k1].conj().T, paap, ra4, optimize=True)
            p1aa = p1aa + pa1a + paa1

            dm2_k = _get_casdm2_kpts(casdm2, mo_phase1, (k1, k2, k3, k4))
            
            g[k1][:, ncore:nocc] += np.einsum('puwx,vuwx->pv', p1aa, dm2_k, optimize=True)

        papa = ppaa = paap = p1aa = paa1 = pa1a = dm2_k = None

        return np.hstack([mc.pack_uniq_var(g[k] - g[k].conj().T) for k in range(nkpts)])    

    # Step-3: Hessian diagonal
    jkcaa = np.zeros((nkpts, nocc, ncas), dtype=dtype)
    # TODO: Should host the hdm2 on the disk.
    hdm2 = np.zeros((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    hdm2_ppmm = np.zeros((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    # This code is benchmarked on the molecular code.
    # ppaa = eris.ppaa
    # papa = eris.papa
    # jtmp = lib.dot(ppaa.reshape(nmo*nmo,-1), casdm2.reshape(ncas*ncas,-1))
    # jtmp = jtmp.reshape(nmo,nmo,ncas,ncas) # nmo, nmo, ncas, ncas
    # ktmp = lib.dot(papa.transpose(0,2,1,3).reshape(nmo*nmo,-1), dm2tmp) # nmo, nmo, ncas, ncas
    # hdm2 = (ktmp.reshape(nmo,nmo,ncas,ncas)+jtmp).transpose(0,2,1,3) # nmo, ncas, nmo, ncas
    # jkcaa  = 6.0 * numpy.einsum('iuiv,uv->iu', papa[:nocc, :, :nocc, :], casdm1)
    # jkcaa -= 2.0 * numpy.einsum('iiuv,uv->iu',ppaa[:nocc, :nocc, :, :], casdm1)

    # hdm2: which is the 2e part of the hessian diagonal. It have the contraction of 2e integrals with 2-RDMs.
    # After a lot of days: I think this is right contractions.
    # hdm2_ref = numpy.einsum('pqwx, wxuv->puqv', ppaa, casdm2)
    # hdm2_ref += numpy.einsum('pwxq,wuxv->puqv', paap, casdm2)
    # hdm2_ref += numpy.einsum('pwxq,uwxv->puqv', paap, casdm2)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        # jkcaa term
        if (k1 == k2 == k3):
            ppaa = eris.ppaa(k1, k2, k3)  # (k1, k2, k3, k4)
            papa = eris.papa(k1, k2, k3)  # (k1, k2, k3, k4)
            for i in range(nocc):
                jkcaa[k1, i] += 6.0/nkpts * np.einsum('uiv,uv->i', papa[i][:, ncore:nocc, :], casdm1_kpts[k1])
                jkcaa[k1, i] -= 2.0/nkpts * np.einsum('iuv,uv->i', ppaa[i][ncore:nocc, :, :], casdm1_kpts[k1])

        # hdm2: K1-term: Debugged  
        # # pwqx(+-+-) uwvx(+-+-) - > puqv (+-+-)
        for kw in range(nkpts):
            kx = kconserv[k1, kw, k3]
            if kconserv[k2,kw,k4] !=kx:
                continue
            papa = eris.papa(k1, kw, k3)
            dm2_blk = casdm2_kpts[k2, kw, k4]
            hdm2[k1, k2, k3] += (1.0 / nkpts) * np.einsum('pwqx,uwvx->pvqu', papa, dm2_blk, optimize=True).transpose(0, 3, 2, 1).conj()
    
    # # pqwx(+-+-), wxvu(+-+-)->puqv(++--)
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k3, k2]
        for kw in range(nkpts):
            kx = kconserv[k1, k3, kw]
            if kconserv[kw, kx, k4] == k2:
                ppaa = eris.ppaa(k1, k3, kw)      # (k1, k3, kw, kx)
                dm2_blk = casdm2_kpts[kw, kx, k4] #.conj() # (kw, kx, k2, k4)
                hdm2_ppmm[k1, k2, k3] += (1.0 / nkpts) * np.einsum('pqwx,wxvu->pquv', ppaa, dm2_blk, optimize=True).transpose(0, 2, 1, 3).conj()
    
    # pwxq(++--) uwxv(+-+-) - > puqv (++--)
    for kp, ku, kq in kpts_helper.loop_kkk(nkpts):
        for kw in range(nkpts):
            kx = kconserv[kp, kq, kw]
            eri_ppmm = eris.paap_ppmm(kp, kw, kx)
            assert kconserv[kp, kq, ku] == kconserv[ku, kw, kx]
            dm2_pmmp = casdm2_kpts[ku, kw, kx]
            hdm2_ppmm[kp, ku, kq] += (1.0 / nkpts) * np.einsum('pwxq,uwxv->pvqu', eri_ppmm, 
                                                            dm2_pmmp, optimize=True).transpose(0, 3, 2, 1).conj()

    # Single determinant limit.
    # hdm2 = np.zeros((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    # I = np.eye(ncas)

    # # pqwx(+-+-), wxvu(+-+-)->puqv(++--)
    # for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
    #     k4 = kconserv[k1, k3, k2]
    #     for kw in range(nkpts):
    #         kx = kconserv[k1, k3, kw]
    #         if kconserv[kw, kx, k4] != k2:
    #             continue
    #         pqwx = eris.ppaa(k1, k3, kw)
    #         if kw==kx and k2==k4:
    #             hdm2_ppmm[k1, k2, k3] += (4.0 / nkpts) * np.einsum('pqww,vu->pqvu', pqwx, I, optimize=True).transpose(0, 3, 1, 2).conj()
    #         if kw==k2 and kx==k4:
    #             hdm2_ppmm[k1, k2, k3] -= (2.0 / nkpts) * np.einsum('pqwx,wu,xv->pquv', pqwx, I, I, optimize=True).transpose(0, 2, 1, 3).conj()

    # # pwqx(+-+-) uwvx(+-+-) - > puqv (+-+-)
    # for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
    #     k4 = kconserv[k1, k2, k3]
    #     for kw in range(nkpts):
    #         kx = kconserv[k1, kw, k3]
    #         if kconserv[k2,kw,k4] !=kx:
    #             continue
    #         if k2==kw and k4==kx:
    #             hdm2[k1, k2, k3] +=  (4.0 / nkpts) *  np.einsum('pwqx,uw,vx->puqv', eris.papa(k1, kw, k3), I, I).conj()
    #         if k2==kx and kw==k4:
    #             hdm2[k1, k2, k3] -=  (2.0 / nkpts) *  np.einsum('pwqx,ux,wv->pvqu', eris.papa(k1, kw, k3), I, I).transpose(0, 3, 2, 1).conj()

    # # pwxq(++--) uwxv(+-+-) - > puqv (++--)
    # for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
    #     k4 = kconserv[k1, k3, k2]
    #     for kw in range(nkpts):
    #         kx = kconserv[k1,k3,kw]
    #         if kconserv[kw,k2,k4] !=kx:
    #             continue
    #         if k2==kw and kx==k4:
    #             hdm2_ppmm[k1, k2, k3] +=  (4.0 / nkpts) *  eris.paap_ppmm(k1, kw, kx).transpose(0,1,3,2).conj()
    #         if kw==kx and k1 == k3 and k2 == k4:
    #             hdm2_ppmm[k1, k2, k3] -= (2.0 / nkpts) * np.einsum('pwwq,uv->puqv', eris.paap_ppmm(k1, kw, kx), I, optimize=True).conj()

    ppaa = papa = jtmp = temp = None

    hdiag = np.zeros((nkpts, nmo, nmo), dtype=dtype)

    for k in range(nkpts):
        temp = np.einsum('ii,jj->ij', h1e_mo[k], dm1[k])
        temp -= h1e_mo[k] * dm1[k]
        hdiag[k] = temp + temp.conj().T

        g_diag = g[k].diagonal() 
        hdiag[k] -= g_diag.conj() + g_diag.reshape(-1, 1)
        idx = np.arange(nmo)
        hdiag[k][idx, idx] += 2.0 * g_diag
        v_diag = vhf_ca[k].diagonal()
        hdiag[k][:, :ncore] += 2.0 * v_diag.reshape(-1, 1)
        hdiag[k][:ncore] += 2.0 * v_diag
        idx = np.arange(ncore)
        hdiag[k][idx, idx] -= 4.0 * v_diag[:ncore]
        tmp = np.einsum('ii,jj->ij', eris.vhf_c[k], casdm1_kpts[k])
        hdiag[k][:, ncore:nocc] += tmp
        hdiag[k][ncore:nocc, :] += tmp.conj().T
        tmp = -eris.vhf_c[k][ncore:nocc,ncore:nocc] * casdm1_kpts[k]
        hdiag[k][ncore:nocc,ncore:nocc] += tmp + tmp.conj().T
    
        # TODO: Remember to divide the eris.j_pc and eris.k_pc by nkpts, 
        # because they are summed over the k-points in the eris generation.
        tmp = 6 * eris.k_pc[k] - 2 * eris.j_pc[k]
        tmp /= nkpts
        hdiag[k][ncore:,:ncore] += tmp[ncore:]
        hdiag[k][:ncore,ncore:] += tmp[ncore:].conj().T
        
        hdiag[k][:nocc,ncore:nocc] -= jkcaa[k]
        hdiag[k][ncore:nocc,:nocc] -= jkcaa[k].conj().T
    
        v_diag = np.einsum('ijij->ij', hdm2[k, k, k])
        hdiag[k][ncore:nocc,:] += v_diag.conj().T
        hdiag[k][:,ncore:nocc] += v_diag

    # Pack the gradients and hessian diagonal    
    g_orb = np.hstack([mc.pack_uniq_var(g[k] - g[k].conj().T) 
                    for k in range(nkpts)])
    h_diag = np.hstack([mc.pack_uniq_var(hdiag[k]) 
                        for k in range(nkpts)])

    # Step-4: Hessian-vector product
    def h_op(x):
        '''
        Compute the Hessian-vector product. Basically, for a given rotation vector x, 
        compute the H*x, which is of the same shape as gradient.
        '''
        # TODO: since the orbital optimization is done for one giant matrix, so
        # I need to unpack this. When I will implement the k-CIAH for the orbital 
        # optimization below won't be required.

        nmopack = mc.pack_uniq_var(np.zeros((nmo, nmo))).shape[0]
        x = np.array([x[i*nmopack:(i+1)*nmopack] 
                    for i in range(nkpts)], dtype=dtype) # (nkpts, nmopack)

        x2 = np.empty((nkpts, nmo, nmo), dtype=dtype)
        np.set_printoptions(precision=3, suppress=True)
        
        if ncore > 0:
            x1 = np.array([mc.unpack_uniq_var(x[k]) 
                        for k in range(nkpts)])
            va, vc = mc.update_jk_in_ah(mo_coeff, x1, casdm1_kpts, eris)
        
        for k in range(nkpts):
            x1 = mc.unpack_uniq_var(x[k].copy()) # (k, k)
            x2[k] = reduce(np.dot, (h1e_mo[k], x1, dm1[k])) # (k, k)
            x2[k] -= 0.5 * np.dot((g[k] + g[k].conj().T), x1) # (k, k)
            x2[k][:ncore] += 2.0 * reduce(np.dot, (x1[:ncore,ncore:], vhf_ca[k][ncore:])) # (k, k)
            x2[k][ncore:nocc] += reduce(np.dot, (casdm1_kpts[k], x1[ncore:nocc], eris.vhf_c[k])) # (k, k)

            # I think this term corresponds to fact that how does the current orbitals will be affected by
            # rotation in some other block.
            for kr in range(nkpts):
                x1temp = mc.unpack_uniq_var(x[kr].copy())
                x2[k][:, ncore:nocc] += np.einsum('purv,rv->pu', hdm2[k, k, kr], x1temp[:, ncore:nocc], optimize=True).conj()
                x2[k][:, ncore:nocc] += np.einsum('purv,pv->ru', hdm2_ppmm[kr, k, k], x1temp[:, ncore:nocc], optimize=True)
               
            if ncore > 0:
                x2[k][ncore:nocc] += va[k]
                x2[k][:ncore,ncore:] += vc[k]
            
        x1temp = va = vc = None
        return np.hstack([mc.pack_uniq_var(x2_ - x2_.conj().T) for x2_ in x2])

    return g_orb, gorb_update, h_op, h_diag

#TODO: make mo_coeff as tagged array then mo_phase can be added to it.
# Currently, plugging all the orbitals together, like done for the kHF.
# More optimum would be do the CIAH separately for each k-point.
def rotate_orb_cc(casscf, mo_coeff, mo_phase, fcivec, fcasdm1, fcasdm2,
                  eris, x0_guess=None, conv_tol_grad=1e-4, max_stepsize=None,
                  verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None: max_stepsize = casscf.max_stepsize
    t3m = (logger.process_clock(), logger.perf_counter())
    u = 1 #[1,]*casscf.nkpts
    g_orb, gorb_update, h_op, h_diag = \
        gen_g_hop(casscf, mo_coeff, mo_phase, u, fcasdm1(), fcasdm2(), eris)
    g_kf = g_orb
    norm_gkf = norm_gorb = np.linalg.norm(g_orb) #np.array([np.linalg.norm(g_orb_) for g_orb_ in g_orb], dtype=g_orb.dtype)
    log.debug('    |g|=%5.3g', np.mean(norm_gorb)) # Mean norm of the orbital gradient
    # log.debug('    max|g|=%5.3g', np.max(norm_gorb)) # Max norm of the orbital gradient (Should print the k-pt as well)
    t3m = log.timer('gen h_op', *t3m)
    
    if norm_gorb < conv_tol_grad * 0.3:
        u = casscf.update_rotate_matrix(g_orb*0)
        yield u, g_orb, 1, x0_guess
        return

    # This is preconditioner for orbital optimization using iterative solver.
    # This preconditioner is defined when CIAH would be solved for each k-point separately.
    # There is preprint on k-CIAH, once that is published and accepted in the main pyscf repo.
    # I will modify the orbital optimization acc to that.
    # def precond(x, e):
    #     hdiagd = np.zeros_like(h_diag)
    #     assert len(x) == len(h_diag)
    #     for k in range(casscf.nkpts):
    #         hdiagd[k] = h_diag[k] - (e - casscf.ah_level_shift)
    #         hdiagd[k][abs(hdiagd[k]) < 1e-8] = 1e-8
    #         x[k] /= hdiagd[k]
    #         norm_x = np.linalg.norm(x[k])
    #         x[k] *= 1/norm_x # Be careful about this. (I mean it can be zero as well.)
    #     hdiagd = None
    #     return x

    def precond(x, e):
        assert x.shape == h_diag.shape
        x = x.copy()
        hdiagd = h_diag.real - (e - casscf.ah_level_shift)
        hdiagd[np.abs(hdiagd) < 1e-8] = 1e-8
        x /= hdiagd
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-14:
            x /= norm_x
        return x
    
    jkcount = 0
    if x0_guess is None:
        x0_guess = g_orb
    
    imic = 0
    dr = 0
    ikf = 0
    
    g_op = lambda: g_orb
    
    problem_size = np.array([np.array(g_orb_).size for g_orb_ in g_orb])
    assert problem_size.sum() == problem_size[0] * len(g_orb)
    problem_size = problem_size.sum()

    for ah_end, ihop, w, dxi, hdxi, residual, seig \
        in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                            tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                            lindep=casscf.ah_lindep, verbose=log):
    
        norm_residual = np.mean([np.linalg.norm(residual_) for residual_ in residual])
        if (ah_end or ihop == casscf.ah_max_cycle or 
            ((norm_residual < casscf.ah_start_tol) and 
             (ihop >= casscf.ah_start_cycle)) or (seig < casscf.ah_lindep)):
            imic += 1
            dxmax = np.max(np.abs(dxi))
            if ihop == problem_size:
                log.debug1('... Hx=g fully converged for small systems')

            elif dxmax > max_stepsize:
                scale = max_stepsize / dxmax
                log.debug1('... scale rotation size %g', scale)
                dxi *= scale
                hdxi *= scale
            
            g_orb = g_orb + hdxi
            dr = dr + dxi
            norm_gorb = np.linalg.norm(g_orb) #np.mean([np.linalg.norm(g_orb_) for g_orb_ in g_orb])
            norm_dxi = np.linalg.norm(dxi)  # np.mean([np.linalg.norm(dxi_) for dxi_ in dxi])
            norm_dr = np.linalg.norm(dr) # np.mean([np.linalg.norm(dr_) for dr_ in dr])

            # These errors are mean-values across the k-points.
            log.debug('    imic %2d(%2d)  |g[o]|=%5.3g  |dxi|=%5.3g  '
                      'max(|x|)=%5.3g  |dr|=%5.3g  eig=%5.3g  seig=%5.3g',
                      imic, ihop, norm_gorb, norm_dxi,
                      dxmax, norm_dr, w, seig)

            ikf += 1
            if (ikf > 1) and (norm_gorb > norm_gkf * casscf.ah_grad_trust_region):
                g_orb = g_orb - hdxi # np.array([g_orb_ - hdxi_ for g_orb_, hdxi_ in zip(g_orb, hdxi)])
                dr -= dxi
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (norm_gorb < 0.3 * conv_tol_grad):
                break

            elif (ikf >= max(casscf.kf_interval, - np.log(norm_dr + 1e-7)) or
                  norm_gorb < norm_gkf/casscf.kf_trust_region):
                ikf = 0
                u = casscf.update_rotate_matrix(dr, u)
                t3m = log.timer('aug_hess in %2d inner iters' % imic, *t3m)
                yield u, g_kf, ihop+jkcount, dxi

                t3m = (logger.process_clock(), logger.perf_counter())


                g_kf1 = gorb_update(u, fcivec())
                jkcount += 1

                norm_gkf1 = np.linalg.norm(g_kf1)
                norm_dg = np.linalg.norm(g_kf1 - g_orb)
                log.debug('    |g|=%5.3g (keyframe), |g-correction|=%5.3g',
                          norm_gkf1, norm_dg)
                
                # For out of trust region
                if (norm_dg > norm_gorb*casscf.ah_grad_trust_region and
                    norm_gkf1 > norm_gkf and
                    norm_gkf1 > norm_gkf*casscf.ah_grad_trust_region):
                    log.debug('    Keyframe |g|=%5.3g  |g_last| =%5.3g out of trust region',
                              norm_gkf1, norm_gorb)
                    
                    dr = -dxi * (1 - casscf.scale_restoration)
                    g_kf = g_kf1
                    break

                t3m = log.timer('gen h_op', *t3m)
                g_orb = g_kf = g_kf1
                norm_gorb = norm_gkf = norm_gkf1
                dr = [np.zeros_like(dr_) for dr_ in dr]
    
    u = casscf.update_rotate_matrix(dr, u)
    yield u, g_kf, ihop+jkcount, dxi


def kernel(casscf, mo_coeff, mo_phase, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''
    Quasi-newton CASSCF optimization driver.
    This is based on CIAH solver of Qiming et. al.
    '''
    log = logger.new_logger(casscf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 1-step CASSCF')
    if callback is None:
        callback = casscf.callback

    if ci0 is None:
        ci0 = casscf.ci

    mo = mo_coeff
    nmo = mo_coeff[0].shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    nkpts = casscf.nkpts
    nelecas = casscf.nelecas

    eris = casscf.ao2mo(mo) # Have to rewrite this.
    e_tot, e_cas, fcivec = casscf.casci(mo, mo_phase, ci0, eris, log, locals())

    # In molecular code, this chunk is commented because macro iterations are needed 
    # when added solvent model. In periodic code, for nmo=ncas condition, my code is crashing due to empty
    # lists of gradients and so on. so I will use this check.
    if ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        else:
            mo_energy = None
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    r0 = None

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, nkpts*ncas, (nelecas[0]*nkpts, nelecas[1]*nkpts))
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    imacro = 0
    dr0 = None
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        imicro = 0
        rota = casscf.rotate_orb_cc(mo, mo_phase, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad*.3, max_stepsize, log)
        
        for u, g_orb, njk, r0 in rota:
            imicro += 1
            norm_gorb = np.linalg.norm(g_orb)
            if imicro == 1:
                norm_gorb0 = norm_gorb
            norm_t = np.linalg.norm(u - np.eye(nkpts*nmo))
            t3m = log.timer('orbital rotation', *t3m)
            if imicro >= max_cycle_micro:
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g',
                          imicro, norm_t, norm_gorb)
                break
            
            # At this stage, U is packed matrix for all k-points, while the mo_coeff are still
            # stored as k-point wise.
            casdm1, casdm2, gci, fcivec = \
                    casscf.update_casdm(mo, u, fcivec, e_cas, eris, locals())
            norm_ddm = np.linalg.norm(casdm1 - casdm1_last)
            norm_ddm_micro = np.linalg.norm(casdm1 - casdm1_prev)
            casdm1_prev = casdm1
            t3m = log.timer('update CAS DM', *t3m)
            
            # I have kept the gradient in the R-space.
            if isinstance(gci, np.ndarray):
                norm_gci = np.linalg.norm(gci)
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%5.3g  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
            else:
                norm_gci = None
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%s  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
            
            if callable(callback):
                callback(locals())
            
            t3m = log.timer('micro iter %2d'%imicro, *t3m)
            if (norm_t < conv_tol_grad or
                (norm_gorb < conv_tol_grad*0.5 and
                 (norm_ddm < conv_tol_ddm*0.4 or norm_ddm_micro < conv_tol_ddm*0.4))):
                break

        rota.close()
        rota = None

        totmicro += imicro
        totinner += njk

        eris = None
        
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        max_offdiag_u = np.abs(np.triu(u, 1)).max()

        if max_offdiag_u < casscf.small_rot_tol:
            small_rot = True
        else:
            small_rot = False

        if not isinstance(casscf, StateAverageMCSCFSolver):
            # I have to code this up:
            if not isinstance(fcivec, np.ndarray):
                fcivec = small_rot
        else:
            newvecs = []
            for subvec in fcivec:
                if not isinstance(subvec, np.ndarray):
                    newvecs.append(small_rot)
                else:
                    newvecs.append(subvec)
            fcivec = newvecs

        mo_phase = get_mo_coeff_k2R(casscf._scf, mo, ncore, ncas)[-1]
        e_tot, e_cas, fcivec = casscf.casci(mo, mo_phase=mo_phase, ci0=fcivec, eris=eris, verbose=log, envs=locals())

        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
        norm_ddm = np.linalg.norm(casdm1 - casdm1_last)
        casdm1_prev = casdm1_last = casdm1
        log.timer('CASCI solver', *t2m)
        t3m = t2m = t1m = log.timer('macro iter %2d'%imacro, *t1m)
        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and norm_gorb0 < conv_tol_grad and
                norm_ddm < conv_tol_ddm and
                (max_offdiag_u < casscf.small_rot_tol or casscf.small_rot_tol == 0)):
            conv = True

        # if dump_chk and casscf.chkfile:
        #     casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('1-step CASSCF converged in %3d macro (%3d JK %3d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %3d macro (%3d JK %3d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = np.diag(-occ)
    else:
        if casscf.natorb:
            raise NotImplementedError('Natural orbital is not implemented for PBC-CASSCF')

    if dump_chk and casscf.chkfile:
        pass
        # TODO: Implement this later.
        #casscf.dump_chk(locals())

    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


# I needed to make a decision here, I could have inherited from bothe pbccasci and mc1step.CASSCF from
# molecular code. But for safety reasons and my inexperience of OOP, I will just inherit from pbccasci.CASBase.
# Look at the description of the attr and other functions.
class PBCCASSCF(casci.PBCCASBASE):

    __doc__ = molCASSCF.__doc__

    # I didn't want to do this, but I don't know if there is any other way to directly use these options
    # from the molecular code.
    max_stepsize = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_max_stepsize', .02)
    max_cycle_macro = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_conv_tol_grad', None)
    ah_level_shift = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
    ah_start_tol = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)

    internal_rotation = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_internal_rotation', False)
    ci_response_space = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ci_response_space', 4)
    ci_grad_trust_region = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ci_grad_trust_region', 3.0)
    with_dep4 = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_with_dep4', False)
    chk_ci = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_chk_ci', False)
    kf_interval = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

    ao2mo_level = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_natorb', False)
    canonicalization = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_sorting_mo_energy', False)
    scale_restoration = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_scale_restoration', 0.5)
    small_rot_tol = getattr(__config__, 'pbc_mcscf_mc1step_CASSCF_small_rot_tol', 0.01)
    extrasym = None
    callback = None

    _keys = {
        'max_stepsize', 'max_cycle_macro', 'max_cycle_micro', 'conv_tol',
        'conv_tol_grad', 'ah_level_shift', 'ah_conv_tol', 'ah_max_cycle',
        'ah_lindep', 'ah_start_tol', 'ah_start_cycle', 'ah_grad_trust_region',
        'internal_rotation', 'ci_response_space', 'ci_grad_trust_region',
        'with_dep4', 'chk_ci', 'kf_interval', 'kf_trust_region',
        'fcisolver_max_cycle', 'fcisolver_conv_tol', 'natorb',
        'canonicalization', 'sorting_mo_energy', 'scale_restoration',
        'small_rot_tol', 'extrasym', 'callback',
        'frozen', 'chkfile', 'fcisolver', 'e_tot', 'e_cas', 'ci', 'mo_coeff',
        'mo_energy', 'converged',
    }

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, frozen=None):
        casci.PBCCASBASE.__init__(self, kmf, ncas, nelecas, ncore)
        self.frozen = frozen
        self.chkfile = self._scf.chkfile
        self.fcisolver.max_cycle = getattr(__config__,
                                           'pbc_mcscf_mc1step_CASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'pbc_mcscf_mc1step_CASSCF_fcisolver_conv_tol', 1e-8)
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

        __getstate__, __setstate__ = lib.generate_pickle_methods(
                excludes=('chkfile', 'callback'))
        
    
    def dump_flags(self, verbose=None):
        mo_coeff_backup = self.mo_coeff.copy()
        self.mo_coeff = self.mo_coeff[0] # Because the dump_flags in molecular code only works for one set of mo_coeff
        molCASSCF.dump_flags(self, verbose)
        self.mo_coeff = mo_coeff_backup
        del mo_coeff_backup
        log = logger.new_logger(self, verbose)
        log.info('nkpts = %d', self.nkpts)
        return self
    
    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        '''
        args:
            mo_coeff: list or np.ndarray (nkpts, nao, nmo) dtype=np.complex128
                Initial guess of the MCSCF problem.
            mo_phase: list or np.ndarray (nkpts, nmo) dtype=np.complex128
                Initial guess of the phase factors for the molecular orbitals.
            ci0: list of np.ndarray, dtype=np.complex128
                Initial guess of the active space CI wavefunction coefficients.
                Note: this should be equal to supercell ci0. which is equals
                to the (nkpts*ncas, nkpts*nelecas[0], nkpts*nelecas[1])
            callback: function, callback(locals()) 
                Some function to called at the end of each micro/macro iteration.
            _kern: function, don't change this.
        returns:
            Five elements, they are
            e_tot: float (np.complex128)
                total energy/nkpts
            e_cas: float (np.complex128)
                Active space CI energy/nkpts
            ci: list of np.ndarray, dtype=np.complex128
                Active space FCI wavefunction coefficients. Note this would be 
                for the supercell, which is equals to the (nkpts*ncas, nkpts*nelecas[0], nkpts*nelecas[1])
            mo_coeff: list of np.ndarray (nkpts, nao, nmo) dtype=np.complex128
                MCSCF canonical orbital coefficients or Natural orbitals within the active space.
                TODO: Natural orbitals are not yet implemented.
            mo_energy: list of np.ndarray (nkpts, nmo) dtype=np.complex128
                MCSCF canonical orbital energies (diagonal elements of general Fock matrix).
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        else: self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback
        if ci0 is None: ci0 = self.ci

        self.check_sanity()
        self.dump_flags()

        mo_phase = get_mo_coeff_k2R(self._scf, mo_coeff, self.ncore, self.ncas)[-1]

        # print('Start 1-step CASSCF optimization')
        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff, mo_phase,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        # This would be for the total energy/nkpts
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot.real)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
    
    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)
    
    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        raise NotImplementedError('MC2step is not implemented for PBC-CASSCF yet')
    
    def casci(self, mo_coeff, mo_phase, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        fcasci = _fake_h_for_fast_casci(self, mo_coeff, mo_phase=mo_phase, eris=eris)
        # The variable/function name calling is weired. Basically, we are calling the kernel from
        # casci module.
        e_tot, e_cas, fcivec = casci.kernel(fcasci, mo_coeff, ci0, log,
                                            envs=envs)
        if not isinstance(e_cas, (float, np.number)):
            raise RuntimeError('Multiple roots are detected in fcisolver.  '
                               'CASSCF does not know which state to optimize.\n'
                               'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        elif np.ndim(e_cas) != 0:
            # This is a workaround for external CI solver compatibility.
            e_cas = e_cas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %#.15g', e_cas)

            if getattr(self.fcisolver, 'spin_square', None):
                try:
                    norb = self.nkpts * self.ncas
                    neleca = self.nkpts * self.nelecas[0]
                    nelecb = self.nkpts * self.nelecas[1]
                    ss = self.fcisolver.spin_square(fcivec, norb, (neleca, nelecb))
                except NotImplementedError:
                    ss = None
            else:
                ss = None

            if 'imicro' in envs:  # Within CASSCF iteration
                if ss is None:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot.real, (e_tot-envs['elast']).real)
                else:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e  S^2 = %.7f',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot.real, (e_tot-envs['elast']).real, ss[0].real)
                if 'norm_gci' in envs and envs['norm_gci'] is not None:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'].real,
                             envs['norm_gci'].real, envs['norm_ddm'].real, envs['max_offdiag_u'].real)
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'].real, envs['norm_ddm'].real, envs['max_offdiag_u'].real)
            else:  # Initialization step
                if ss is None:
                    log.info('CASCI E = %#.15g', e_tot.real)
                else:
                    log.info('CASCI E = %#.15g  S^2 = %.7f', e_tot.real, ss[0].real)
        return e_tot, e_cas, fcivec

    # casci = molCASSCF.casci
    uniq_var_indices = molCASSCF.uniq_var_indices

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff[0].shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]
    
    def unpack_uniq_var(self, v, hermi=2):
        '''
        Unpack the unique variables into a full matrix.
        hermi: int
            1: Hermitian
            2: Anti-Hermitian
        '''
        v = np.asarray(v)
        nmo = self.mo_coeff[0].shape[1]
        nkpts = self.nkpts
        dtype = self.mo_coeff[0].dtype
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        uniq_idx = int(np.count_nonzero(idx))

        # Decide whether the input is for a single k-point or for all k-points.
        assert v.size == uniq_idx or v.size == self.nkpts * uniq_idx

        def _unpack_uniq_var(v, hermi=2):
            # For a single k-point.
            mat = np.zeros((nmo,nmo), dtype=dtype)
            mat[idx] = v.astype(dtype)
            if hermi == 1:
                return mat + mat.conj().T
            elif hermi == 2:
                return mat - mat.conj().T

        if v.size == uniq_idx:
            return _unpack_uniq_var(v, hermi=hermi)

        elif v.size == nkpts * uniq_idx:
            mats = np.zeros((nkpts, nmo, nmo), dtype=dtype)
            for k in range(nkpts):
                p0 = k * uniq_idx
                p1 = (k + 1) * uniq_idx
                mats[k] = _unpack_uniq_var(v[p0:p1], hermi=hermi)
            return scipy.linalg.block_diag(*mats)

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return np.dot(u0, expmat(dr))
    
    gen_g_hop = gen_g_hop
    rotate_orb_cc = rotate_orb_cc
    get_h2eff = casciModule.PBCCASCI.get_h2eff
    
    def ao2mo(self, mo_coeff):
        '''
        In pbc_casscf, I don't have worry about two options (DF vs non DF) as in the molecular code. 
        The integrals will always be transformed to the MO basis using DF only.
        '''
        eris = _ERIS(self, mo_coeff, method='direct')
        return eris
    
    def update_jk_in_ah(self, mo_coeff, r_k, casdm1_k, eris):
        '''
        Update the J and K matrices in the auxiliary Hamiltonian.
        Using the rotation matrix, rotate the mo_coeff then construct the density matrix
        from that get the potential and then rotate those potential back to mo_basis.
        '''
        cell = self._scf.cell
        kpts = self._scf.kpts
        nkpts = self.nkpts
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        
        # Make sure they are for a single k-point.
        assert casdm1_k.ndim == 3 and casdm1_k.shape[0] == nkpts
        assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == nkpts

        mo_k = np.array(mo_coeff).copy() # (nkpts, nao, nmo)

        def _get_jk_core_or_act(dm_k):
            vj, vk = self._scf.get_jk(cell, dm_k, kpts=kpts, hermi=1, with_j=True,
                                 with_k=True, exxdiv=None)
            return vj, vk
        
        dm3temp = np.array([reduce(np.dot, (mo_k[k][:,:ncore], 
                                            (r_k[k][:ncore,ncore:] @ mo_k[k][:,ncore:].conj().T))) 
                                            for k in range(nkpts)]) # (nkpts, nao, nao)
        dm3 = np.array([dm3temp[k] + dm3temp[k].conj().T 
                        for k in range(nkpts)])

        dm4temp = np.array([reduce(np.dot, (mo_k[k][:,ncore:nocc], casdm1_k[k], 
                                        (r_k[k][ncore:nocc] @ mo_k[k].conj().T))) 
                                        for k in range(nkpts)]) # (nkpts, nao, nao)
        
        dm4 = np.array([dm4temp[k] + dm4temp[k].conj().T 
                        for k in range(nkpts)])
        
        vj, vk  = _get_jk_core_or_act(dm3)
        va = np.array([reduce(np.dot, (casdm1_k[k], (mo_k[k][:,ncore:nocc].conj().T @ 
                                                     (vj[k] * 2.0 - vk[k]) @ mo_k[k]))) 
                                                     for k in range(nkpts)])

        vj, vk = _get_jk_core_or_act(dm3*2.0 + dm4)
        vc = np.array([reduce(np.dot, (mo_k[k][:,:ncore].conj().T, vj[k]*2.0 - vk[k], mo_k[k][:,ncore:]))
                       for k in range(nkpts)])
        return va, vc
 
    def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
        np.set_printoptions(precision=3, suppress=True)
        nkpts = self.nkpts
        kpts = self._scf.kpts
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nmo = mo[0].shape[1]
        nocc = ncore + ncas
        dtype = mo[0].dtype

        u = block_diag_to_kblocks(u, nkpts, nmo) # (nkpts, nmo, nmo)
        
        rmat = np.array([umat - np.eye(nmo) for umat in u], dtype=dtype) # (nkpts, nmo, nmo)

        hcore = self.get_hcore()
        ddm = np.empty((nkpts, nmo, nmo), dtype=dtype)

        h1e_mo = np.empty((nkpts, nmo, nmo), dtype=dtype)
        for k in range(nkpts):
            uc = u[k][:,:ncore]
            h1e_mo[k] = reduce(np.dot, (mo[k].conj().T, hcore[k], mo[k]))
            ddm[k] = np.dot(uc, uc.conj().T) * 2.0
            ddm[k][np.diag_indices(ncore)] -= 2.0
        
        if self.with_dep4:
            mo1 = np.array([np.dot(mo[k], u[k]) 
                            for k in range(nkpts)])
           
            dm_core = np.array([np.dot(mo1[k][:,:ncore], mo1[k][:,:ncore].conj().T) * 2.0
                                 for k in range(nkpts)])
            vj, vk = self._scf.get_jk(self._scf.cell, dm_core)

            mo_phase1 = get_mo_coeff_k2R(self._scf, mo1, ncore, ncas)[-1]

            # h1e for active space.
            h1 = np.empty((nkpts, ncas, ncas), dtype=dtype)
            for k in range(nkpts):
                ua = u[k][:,ncore:nocc].copy()
                mo1_cas = mo1[k][:,ncore:nocc].copy()
                # update h1e for active space in k-space (mo basis)
                h1[k] = reduce(np.dot, (ua.conj().T, h1e_mo[k], ua))
                h1[k] += reduce(np.dot, (mo1_cas.conj().T, vj[k] - vk[k] * 0.5, mo1_cas)) # add the contribution from the vj vk terms
                # do the transformation to R-space (mo basis)
            
            h1_R = lib.einsum('xui,xuv,xvj->ij', mo_phase1.conj(), h1, mo_phase1)
            h2_R = self._exact_paaa(mo, u)
            vj = vk = h1 = None
        
        else:
            # h1e
            mo1 = np.array([np.dot(mo[k], u[k]) 
                            for k in range(nkpts)])
            
            mo_phase1 = get_mo_coeff_k2R(self._scf, mo1, ncore, ncas)[-1]
            
            h1 = np.empty((nkpts, ncas, ncas), dtype=dtype)
            kconserv = kpts_helper.get_kconserv(self._scf.cell, kpts)
            '''
            # I need to loop over the k-points.
            Let me write without loop first.

            jk = reduce(numpy.dot, (ua.T, eris.vhf_c, ua))
            jk += np.einsum('pquv,pq->uv', eris.ppaa, ddm)
            jk -= 0.5 * np.einsum('puqv,pq->uv', eris.papa, ddm)
            '''
            for k in range(nkpts):
                ua = u[k][:,ncore:nocc].copy()
                jk = reduce(np.dot, (ua.conj().T, eris.vhf_c[k], ua))
                # for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
                #     if k1 != k:
                #         continue
                #     else:
                #         ppaa = eris.ppaa(k1, k1, k1) # (k1, k1, k1, k1)
                #         jk += np.einsum('pquv,pq->uv', ppaa, ddm[k1]) # (k1, k1)
                #         papa = eris.papa(k1, k1, k1) # (k1, k1, k1, k1)
                #         jk -= 0.5 * np.einsum('puqv,pq->uv', papa, ddm[k1]) # (k1, k1)
                # I have rewritten above code here:
                ppaa = eris.ppaa(k, k, k) # (k, k, k, k)
                jk += np.einsum('pquv,pq->uv', ppaa, ddm[k]) # (k, k)
                papa = eris.papa(k, k, k) # (k, k, k, k)
                jk -= 0.5 * np.einsum('puqv,pq->uv', papa, ddm[k]) # (k, k)
                
                h1[k] = reduce(np.dot, (ua.conj().T, h1e_mo[k], ua)) # k-space (mo basis)
                h1[k] += jk # k-space (mo-basis)
                
            h1_R = lib.einsum('xui,xuv,xvj->ij', mo_phase1.conj(), h1, mo_phase1) # transform to R-space basis
            
            '''
            ppaa = np.einsum('ps, qt, pquv -> stuv', ua, ua, eris.ppaa)
            papa = np.einsum('ps, qt, puqv -> sutv', ua, ua, eris.papa)
            '''
            mo_ks = mo_phase1[kconserv]

            def _convert_to_R_space(eri_k):
                out = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                            mo_phase1.conj(), mo_phase1, eri_k, mo_phase1.conj(), 
                            mo_ks, optimize=True)
                out *= 1.0/nkpts
                return out
            
            aa11 = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=np.complex128)
            aaaa = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=np.complex128)
            
            for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
                ppaa = eris.ppaa(k1, k2, k3)
                aaaa[k1, k2, k3] = ppaa[ncore:nocc,ncore:nocc,:,:].copy()
                aa11[k1, k2, k3] = np.einsum('ps, qt, pquv -> stuv', u[k1][:,ncore:nocc], 
                                             u[k2][:,ncore:nocc], ppaa)
        
            aa11_R = _convert_to_R_space(aa11)
            aaaa_R = _convert_to_R_space(aaaa)
            
            aa11 = aaaa = None

            aa11_R = aa11_R + aa11_R.conj().transpose(2,3,0,1) - aaaa_R


            a11a = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=np.complex128)
            for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
                papa = eris.papa(k1, k2, k3)
                a11a[k1, k2, k3] = np.einsum('ps, qt, puqv -> sutv', rmat[k1][:,ncore:nocc], rmat[k3][:,ncore:nocc], papa)

            a11a_R = _convert_to_R_space(a11a)

            a11a = None

            a11a_R = a11a_R + a11a_R.conj().transpose(1, 0, 2, 3)
            a11a_R = a11a_R + a11a_R.conj().transpose(0, 1, 3, 2)
            
            h2_R = aa11_R + a11a_R
            aa11_R = a11a_R = None

        ecore = 0
        for k in range(nkpts):
            ecore += np.einsum('pq,pq->', h1e_mo[k], ddm[k])
            ecore += np.einsum('pq,pq->', eris.vhf_c[k], ddm[k])
        ecore += self.energy_nuc() * nkpts

        ci1, g = self.solve_approx_ci(h1_R, h2_R, fcivec, ecore, e_cas, envs)
        # In case of external CI solvers like DMRG, or even for the state-average condition
        # we won't need this.
        if g is not None:
            ovlp = np.vdot(fcivec.ravel(), ci1.ravel())
            norm_g = np.linalg.norm(g)
            if (1 - abs(ovlp) > norm_g) * self.ci_grad_trust_region:
                logger.debug(self, '<ci1|ci0>=%5.3g |g|=%5.3g, ci1 out of trust region',
                             ovlp, norm_g)
                ci1 = fcivec.ravel() + g
                ci1 *= 1/np.linalg.norm(ci1)
        
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, nkpts*ncas, (nelecas[0]*nkpts, nelecas[1]*nkpts))

        return casdm1, casdm2, g, ci1
        
    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs):
        '''
        Solving the CI eigenvalue or response problem approximately.
        Code is adapted from the molecular code, expect few datatype and sanity checks.
        '''
        ncas = self.ncas
        nelecas = self.nelecas
        nkpts = self.nkpts
        dtype = h1.dtype

        if 'norm_gorb' in envs: tol = max(self.conv_tol, envs['norm_gorb']**2 * 0.1)
        else: tol = None

        if getattr(self.fcisolver, 'approx_kernel', None):
            raise NotImplementedError('approx_kernel is not tested/implemented for direct_spin1_cplx')
        elif not (getattr(self.fcisolver, 'contract_2e', None) and 
                  getattr(self.fcisolver, 'absorb_h1e', None)):
            raise NotImplementedError('direct kernel is not tested/implemented for direct_spin1_cplx')

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas*nkpts, (nkpts*nelecas[0], nkpts*nelecas[1]), 0.5)

        def contract_2e(c):
            hc = self.fcisolver.contract_2e(h2eff, c, ncas*nkpts, (nelecas[0]*nkpts, nelecas[1]*nkpts))
            return hc.ravel()

        e_ci = e_cas - ecore

        hc = contract_2e(ci0)

        g = hc - e_ci * ci0.ravel()

        if self.ci_response_space > 7 or ci0.size <= self.fcisolver.pspace_size:
            logger.debug(self, 'CI step by full response')
            max_memory = max(400, self.max_memory - lib.current_memory()[0])
            e, ci1 = self.fcisolver.kernel(h1, h2, nkpts*ncas, (nelecas[0]*nkpts, nelecas[1]*nkpts), ecore=ecore,
                                           ci0=ci0, tol=tol, max_memory=max_memory)
        else:
            nd = min(self.ci_response_space, ci0.size)
            xs = [ci0.ravel().copy()]
            ax = [hc.copy()]
            heff = np.empty((nd, nd), dtype=dtype)
            seff = np.empty((nd, nd), dtype=dtype)

            heff[0,0] = np.vdot(xs[0], ax[0])
            seff[0,0] = np.vdot(xs[0], xs[0])

            tol_residual = self.fcisolver.conv_tol ** 0.5

            for i in range(1, nd):
                dx = ax[i-1] - e_ci * xs[i-1]
                if np.linalg.norm(dx) < tol_residual:
                    break

                xs.append(dx)
                ax.append(contract_2e(xs[i]))

                for j in range(i+1):
                    hij = np.vdot(xs[i], ax[j])
                    sij = np.vdot(xs[i], xs[j])
                    heff[i,j] = hij
                    heff[j,i] = hij.conj()
                    seff[i,j] = sij
                    seff[j,i] = sij.conj()

            nd = len(xs)
            e, v, seig = lib.safe_eigh(heff[:nd, :nd], seff[:nd, :nd])

            ci1 = np.zeros_like(xs[0], dtype=np.result_type(*xs, v))
            for i in range(nd):
                ci1 += xs[i] * v[i, 0]
            ci1 = ci1.reshape(ci0.shape)
        return ci1, g
    
    def _gen_g_hop_test(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if eris is None: eris = self.ao2mo(mo_coeff)
        # The mo_phase is needed for the transformation of the integrals to R-space. 
        # I will just compute it here.
        mo_phase = get_mo_coeff_k2R(self._scf, mo_coeff, self.ncore, self.ncas)[-1]
        if casdm1_casdm2 is None:
            nkpts = self.nkpts
            ncastot = nkpts * self.ncas
            nelecas = (nkpts * self.nelecas[0], nkpts * self.nelecas[1])
            civec = self.casci(mo_coeff, mo_phase, self.ci, eris)[2]
            casdm1, casdm2 = self.fcisolver.make_rdm12(civec, ncastot, nelecas)
        else:
            casdm1, casdm2 = casdm1_casdm2
        return self.gen_g_hop(mo_coeff, mo_phase, 1, casdm1, casdm2, eris)

    def get_grad(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        return self._gen_g_hop_test(mo_coeff, casdm1_casdm2=casdm1_casdm2, eris=eris)[0]

    def get_grad_update(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        return self._gen_g_hop_test(mo_coeff, casdm1_casdm2=casdm1_casdm2, eris=eris)[1]

    def get_hessian_op(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        return self._gen_g_hop_test(mo_coeff, casdm1_casdm2=casdm1_casdm2, eris=eris)[2]
    
    def get_hessian_diag(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        return self._gen_g_hop_test(mo_coeff, casdm1_casdm2=casdm1_casdm2, eris=eris)[3]

    def _exact_paaa(self, mo_kpts, u_kpts, out=None):
        '''
        # In the molecular code, the paaa term is created which is then
        # sliced to get the aaaa. Instead, I will directly compute the aaaa term here.
        # Note: I kept the same function name as in the molecular code.
        '''
        kmf = self._scf
        cell = kmf.cell
        kpts = kmf.kpts
        nkpts = self.nkpts
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        mo1 = [np.dot(mo, u) for mo, u in zip(mo_kpts, u_kpts)]
        mo_phase1 = get_mo_coeff_k2R(kmf, mo1, ncore, ncas)[-1]
        mo_cas_kpts = np.array([mo1[k][:, ncore:nocc] for k in range(nkpts)])
        eri_k = kmf.with_df.ao2mo_7d(mo_cas_kpts, kpts=kpts)
        kconserv = kpts_helper.get_kconserv(cell, kpts)
        mo_ks = mo_phase1[kconserv]
        
        aaaa = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                         mo_phase1.conj(), mo_phase1, eri_k, mo_phase1.conj(), mo_ks, optimize=True)
        aaaa *= 1.0/nkpts

        eri_k = mo_ks = kconserv = None 

        ncastot = nkpts * ncas
        assert aaaa.shape == (ncastot, ncastot, ncastot, ncastot)
        return aaaa
    
    def dump_chk(self, **kwargs):
        pass

    def update_from_chk(self, chkfile=None):
        raise NotImplementedError('update_from_chk is not implemented for PBC-CASSCF yet')
    
    update = update_from_chk

    def rotate_mo(self, mo_coeff, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        nmo = mo_coeff[0].shape[1]
        nkpts = self.nkpts
        u = block_diag_to_kblocks(u, self.nkpts, nmo)
        for k in range(nkpts):
            mo_coeff[k] = np.dot(mo_coeff[k], u[k])
        if log is not None and log.verbose >= logger.DEBUG:
            ncore = self.ncore
            ncas = self.ncas
            nocc = ncore + ncas
            ovlp = self._scf.get_ovlp()
            for k in range(nkpts):
                log.debug('K-point %d', k)
                s = reduce(np.dot, (mo_coeff[k][:,ncore:nocc].conj().T, ovlp[k],
                                    self.mo_coeff[k][:,ncore:nocc]))
                log.debug('Active space overlap to initial guess, SVD = %s',
                        np.linalg.svd(s)[1])
                log.debug('Active space overlap to last step, SVD = %s',
                        np.linalg.svd(u[k][ncore:nocc,ncore:nocc])[1])
        return mo_coeff
    
    micro_cycle_scheduler = molCASSCF.micro_cycle_scheduler
    max_stepsize_scheduler = molCASSCF.max_stepsize_scheduler
    ah_scheduler = molCASSCF.ah_scheduler

    # I don't know, whether these can be imported or assigned directly from the molecular code
    # but I will just write them here for now. I will try to import them later.

    @property
    def max_orb_stepsize(self):
        return self.max_stepsize
    
    @max_orb_stepsize.setter
    def max_orb_stepsize(self, x):
        sys.stderr.write('WARN: Attribute "max_orb_stepsize" was replaced by "max_stepsize"\n')
        self.max_stepsize = x
    
    @property
    def ci_update_dep(self):
        return self.with_dep4
    
    @ci_update_dep.setter
    def ci_update_dep(self, x):
        sys.stderr.write('WARN: Attribute .ci_update_dep was replaced by .with_dep4 since PySCF v1.1.\n')
        self.with_dep4 = x == 4

    grad_update_dep = ci_update_dep

    @property
    def max_cycle(self):
        return self.max_cycle_macro
    
    @max_cycle.setter
    def max_cycle(self, x):
        self.max_cycle_macro = x

    def approx_hessian(self, *args, **kwargs):
        raise NotImplementedError('Approximate Hessian is not implemented for PBC-CASSCF yet')
    
    def nuc_grad_method(self, **args):
        raise NotImplementedError('Nuclear gradient method is not implemented for PBC-CASSCF yet')
    
    def _state_average_nuc_grad_method (self, **args):
        raise NotImplementedError('State-average nuclear gradient method is not implemented for PBC-CASSCF yet')
    
    def _state_average_nac_method(self):
        raise NotImplementedError('State-average NAC method is not implemented for PBC-CASSCF yet')
    
    def newton(self):
        raise NotImplementedError('Newton solver is not implemented for PBC-CASSCF yet')
    
    def reset(self, cell=None):
        casci.CASBase.reset(self, cell=cell)
        self._max_stepsize = None

CASSCF = PBCCASSCF

def expmat(a):
    # Should import this.
    return scipy.linalg.expm(a)

def block_diag_to_kblocks(mat, nkpts, nmo):
    # This is helper function convert the block diagonal matrix to a list of k-blocks. 
    # Example U matrix packed as (8,8) in two blocks for two k-points, and 
    # I want to convert it to two (4,4) matrices for each k-point.
    return np.array([mat[k*nmo:(k+1)*nmo, k*nmo:(k+1)*nmo]
                     for k in range(nkpts)])

from mrh.my_pyscf.pbc.mcscf.casci import PBCCASCI

def _fake_h_for_fast_casci(casscf, mo, mo_phase, eris):
    mc = casscf.view(PBCCASCI)
    mc.mo_coeff = mo
    mc.mo_phase = mo_phase
    if eris is None:
        return mc

    cell = casscf._scf.cell
    kpts = casscf._scf.kpts
    nkpts = casscf.nkpts
    assert len(kpts) == nkpts
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas
    ncas = casscf.ncas
    dtype = mo[0].dtype

    # Core energy contribution
    mo_core = [mo[k][:,:ncore] for k in range(nkpts)]
    mo_cas = [mo[k][:,ncore:nocc] for k in range(nkpts)]
    core_dm = [np.dot(mo_core[k], mo_core[k].conj().T) * 2 
               for k in range(nkpts)]
    
    hcore = casscf.get_hcore()
    energy_core = nkpts * casscf.energy_nuc() # Remember energy would be divided by nkpts in the end.
    energy_core += sum([np.einsum('ij,ji', core_dm[k], hcore[k]) 
                        for k in range(nkpts)])
    energy_core += sum([eris.vhf_c[k][:ncore,:ncore].trace() 
                        for k in range(nkpts)])
    # h1, and h2 in mo basis
    h1eff = np.asarray([reduce(np.dot, (mo_cas[k].conj().T, hcore[k], mo_cas[k])) 
                        for k in range(nkpts)])
    h1eff += np.asarray([eris.vhf_c[k][ncore:nocc,ncore:nocc] 
                         for k in range(nkpts)])
    h1eff = lib.einsum('xui,xuv,xvj->ij', mo_phase.conj(), h1eff, mo_phase)

    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    kconserv = kpts_helper.get_kconserv(cell, kpts)
    mo_ks = mo_phase[kconserv]
    eri_k = np.empty((nkpts,nkpts,nkpts, ncas, ncas, ncas, ncas), dtype=dtype)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        eri_k[k1,k2,k3] = eris.ppaa(k1,k2,k3)[ncore:nocc,ncore:nocc, :, :]
    
    eri_cas = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                         mo_phase.conj(), mo_phase, eri_k, mo_phase.conj(), mo_ks, optimize=True)
    eri_cas *= 1.0/nkpts
    mc.get_h2eff = lambda *args: eri_cas
    return mc

if __name__ == '__main__':
    pass