import os
import sys
import numpy as np
from functools import reduce

from pyscf import lib, __config__
from pyscf.soscf import ciah # Recently they have added the CIAH solver for PBC. Will use it!
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.pbc.lib import kpts_helper
from pyscf.mcscf.mc1step import max_stepsize_scheduler
logger = lib.logger

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
        g = np.zeros_like((nkpts, nmo, nmo), dtype=dtype)
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
        for k in range(nkpts):
            uc = u[k][:, :ncore].copy()
            ua = u[k][:, ncore:nocc].copy()
            rmat = u[k] - np.eye(nmo)
            ra = rmat[:, ncore:nocc].copy()
            mo1 = np.dot(mo_coeff[k], u[k])
            mo_c = np.dot(mo_coeff[k], uc)
            mo_a = np.dot(mo_coeff[k], ua)
            dm_c = np.dot(mo_c, mo_c.conj().T) * 2.0
            casdm1_k = reduce(np.dot, (mo_phase[k], casdm1, mo_phase[k].conj().T))
            dm_a = reduce(np.dot, (mo_a, casdm1_k, mo_a.conj().T))
            vj, vk = mc.get_jk(mc._scf.cell, (dm_c, dm_a), kpts=mc._scf.kpts)
            vhf_c = reduce(np.dot, (mo1.conj().T, vj[0]-vk[0]*0.5, mo1[:,:nocc]))
            vhf_a = reduce(np.dot, (mo1.conj().T, vj[1]-vk[1]*0.5, mo1[:,:nocc]))
            h1e_mo1 = reduce(np.dot, (u[k].conj().T, h1e_mo[k], u[k][:,:nocc]))
            
            p1aa = np.empty((nkpts, nkpts, nkpts, nmo, ncas, ncasncas), dtype=dtype)
            paa1 = np.empty((nkpts, nkpts, nkpts, nmo, ncasncas, ncas), dtype=dtype)
            aaaa = np.empty((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=dtype)

            for k1, k2, k3 in kpts_helper.loop(nkpts):
                k4 = kconserv[k1, k2, k3]
                if not k4 == k:
                    pass
                else:
                    jbuf = eris.ppaa[k1, k2, k3]
                    kbuf = eris.papa[k1, k2, k3]
                    p1aa[k1, k2, k3] = np.einsum('pu, pqm-> qum', ua, jbuf.reshape(nmo, nmo, -1))
                    paa1[k1, k2, k3] = np.einsum('pqm, pu-> qmu', kbuf.conj().transpose(0,2,1,3).reshape(nmo, nmo, -1), ra)
                    aaaa[k1, k2, k3] = jbuf[ncore:nocc, ncore:nocc, :, :]
                
            g[k][:, :ncore] = 2.0 * (h1e_mo1[k][:,:ncore] + vhf_c[k][:,:ncore] + vhf_a[k][:,:ncore])
            g[k][:,ncore:nocc] = np.dot(h1e_mo1[k][:,ncore:nocc] + vhf_c[k][:,ncore:nocc], casdm1_k)

            for k1, k2, k3 in kpts_helper.loop(nkpts):
                k4 = kconserv[k1, k2, k3]
                if not k4 == k:
                    pass
                else:
                    dm2_k = np.einsum('ip, jq, pqrs, kr, ls->ijkl', 
                                      mo_phase[k1], mo_phase[k2], casdm2, mo_phase[k3], mo_phase[k4])
                    p1aa_k = lib.dot(u[k].conj().T, p1aa[k1, k2, k3].reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
                    paa1_k = lib.dot(u[k].conj().T, paa1[k1, k2, k3].reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
                    p1aa_k += paa1_k
                    p1aa_k += paa1_k.conj().transpose(0, 1, 3, 2)
                    g[k][:, ncore:nocc] += np.einsum('puwx, wxuv->pv', p1aa_k, dm2_k)

        return [mc.pack_uniq_var(grd - grd.conj().T) for grd in g]    
    
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
        x2 = np.empty((nkpts, nmo, nmo), dtype=dtype)
        for k in range(nkpts):
            x1 = mc.unpack_uniq_var(x[k])
            x2[k] = reduce(np.dot, (h1e_mo[k], x1, dm1[k]))
            x2[k] -= 0.5 * np.dot((g[k] + g[k].conj().T), x1)
            x2[k][:ncore] += 2.0 * reduce(np.dot, (x1[:ncore,ncore:], vhf_ca[k][ncore:]))
            x2[k][ncore:nocc] += reduce(np.dot, (casdm1, x1[ncore:nocc], eris.vhf_c[k]))

            for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
                k4 = kconserv[k1, k2, k3]
                if not k4 == k:
                    continue
                else:
                    x2[k][:, ncore:nocc] += np.einsum('purv,rv->pu', hdm2[k1, k2, k3], x1[:,ncore:nocc])
            
            if ncore > 0:
                # I need to modify this function: mc.update_jk_in_ah as well.
                va, vc = mc.update_jk_in_ah(mo_coeff, x1, casdm1, eris)
                x2[k][ncore:nocc] += va
                x2[k][:ncore,ncore:] += vc
            
            x2[k] = x2[k] - x2[k].conj().T

        return [mc.pack_uniq_var(x2_) for x2_ in x2]
    
    return g_orb, gorb_update, h_op, h_diag


def rotate_orb_cc(casscf, mo_coeff, mo_phase, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None: max_stepsize = casscf.max_stepsize
    t3m = (logger.process_clock(), logger.perf_counter())
    u = [1,]*casscf.nkpts
    g_orb, gorb_update, h_op, h_diag = \
        gen_g_hop(casscf, mo_coeff, mo_phase, u, fcasdm1, fcasdm2, eris)
    
    g_kf = g_orb
    norm_gkf = norm_gorb = np.array([np.linalg.norm(g_orb_) for g_orb_ in g_orb])
    log.debug('    |g|=%5.3g', np.mean(norm_gorb)) # Mean norm of the orbital gradient
    log.debug('    max|g|=%5.3g', np.max(norm_gorb)) # Max norm of the orbital gradient (Should print the k-pt as well)
    t3m = log.timer('gen h_op', *t3m)
    
    if all(norm_gorb < conv_tol_grad*.3):
        u = casscf.update_rotate_matrix(g_orb*0)
        yield u, g_orb, 1, x0_guess
        return
    
    def precond(x, e):
        hdiagd = np.zeros_like(h_diag)
        for k in range(casscf.nkpts):
            hdiagd[k] = h_diag - (e - casscf.ah_level_shift)
            hdiagd[k][abs(hdiagd[k]) < 1e-8] = 1e-8
            x[k] /= hdiagd[k]
            norm_x = np.linalg.norm(x[k])
            x[k] *= 1/norm_x # Be careful about this. (I mean it can be zero as well.)
        hdiagd = None
        return x
    
    jkcount = 0
    if x0_guess is None:
        x0_guess = g_orb
    
    imic = 0
    dr = 0
    ikf = 0
    
    g_op = lambda: g_orb
    problem_size = np.array(g_orb).size

    for ah_end, ihop, w, dxi, hdxi, residual, seig \
        in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                            tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                            lindep=casscf.ah_lindep, verbose=log):
    
        norm_residual = np.mean([np.linalg.norm(residual_) for residual_ in residual])
        if (ah_end or ihop == casscf.ah_max_cycle or 
            ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or 
            (seig < casscf.ah_lindep)):
            imic += 1
            dxmax = np.max(abs(dxi))
            if ihop == problem_size:
                log.debug1('... Hx=g fully converged for small systems')

            elif dxmax > max_stepsize:
                scale = max_stepsize / dxmax
                log.debug1('... scale rotation size %g', scale)
                dxi *= scale
                hdxi *= scale
            
            g_orb = g_orb + hdxi
            dr = dr + dxi
            norm_gorb = np.mean([np.linalg.norm(g_orb_) for g_orb_ in g_orb])
            norm_dxi = np.mean([np.linalg.norm(dxi_) for dxi_ in dxi])
            norm_dr = np.mean([np.linalg.norm(dr_) for dr_ in dr])

            # These errors are mean-values across the k-points.
            log.debug('    imic %2d(%2d)  |g[o]|=%5.3g  |dxi|=%5.3g  '
                      'max(|x|)=%5.3g  |dr|=%5.3g  eig=%5.3g  seig=%5.3g',
                      imic, ihop, norm_gorb, norm_dxi,
                      dxmax, norm_dr, w, seig)

            ikf += 1
            if (ikf > 1) and (norm_gorb > norm_gkf * casscf.ah_grad_trust_region):
                g_orb = np.array([g_orb_ - hdxi_ for g_orb_, hdxi_ in zip(g_orb, hdxi)])
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
                norm_gkf1 = np.mean([np.linalg.norm(g_kf1_) for g_kf1_ in g_kf1])
                norm_dg = np.mean([np.linalg.norm(g_kf1_ - g_orb_) 
                                   for g_kf1_, g_orb_ in zip(g_kf1, g_orb)])
                log.debug('    |g|=%5.3g (keyframe), |g-correction|=%5.3g',
                          norm_gkf1, norm_dg)
                
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
                dr = [0 for _ in dr]
    
    u = casscf.update_rotate_matrix(dr, u)
    yield u, g_kf, ihop+jkcount, dxi


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
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
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())

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
        rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad*.3, max_stepsize, log)
        
        for u, g_orb, njk, r0 in rota:
            imicro += 1
            norm_gorb = np.mean([np.linalg.norm(g_orb_) for g_orb_ in g_orb])
            if imicro == 1:
                norm_gorb0 = norm_gorb
            norm_t = np.mean([np.linalg.norm(u_-np.eye(nmo)) for u_ in u])
            t3m = log.timer('orbital rotation', *t3m)
            if imicro >= max_cycle_micro:
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g',
                          imicro, norm_t, norm_gorb)
                break

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

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())

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

        if dump_chk and casscf.chkfile:
            casscf.dump_chk(locals())

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
        casscf.dump_chk(locals())

    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy