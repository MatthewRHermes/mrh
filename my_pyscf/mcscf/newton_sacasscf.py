#!/usr/bin/env python
# MRH: I copied and started modifying this on 04/04/2019
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Second order CASSCF
'''

import sys
import time
import copy
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import casci, mc1step
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import chkfile
from pyscf import ao2mo
from pyscf import scf
from pyscf.soscf import ciah
from pyscf import fci

# MRH: I keep writing "np" so I'll just set it here so I don't have to worry about it in either direction
np = numpy
from mrh.my_pyscf.mcscf import sacasscf

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, ci0, eris, verbose=None):
# MRH: weights need to be accessible to this function
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nelecas = casscf.nelecas
    nmo = mo.shape[1]
    nroot = len (ci0)
    ndet = ci0[0].size
    weights_1d = np.asarray (casscf.weights) # To facilitate broadcasting
    weights_2d = np.asarray (casscf.weights)[:,None] 
    weights_3d = np.asarray (casscf.weights)[:,None,None] 
    # MRH: contiguous in memory but two-dimensional
    ci0 = np.ravel (np.asarray (ci0)).reshape (nroot, -1)

    if getattr(casscf.fcisolver, 'gen_linkstr', None):
        linkstrl = casscf.fcisolver.gen_linkstr(ncas, nelecas, True)
        linkstr  = casscf.fcisolver.gen_linkstr(ncas, nelecas, False)
    else:
        linkstrl = linkstr  = None
    def fci_matvec(civec, h1, h2):
        if civec.ndim == 1: civec=civec[None,:] # Edge case
        # MRH: contract all ci vectors
        h2cas = casscf.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)
        # MRH: contiguous in memory but two-dimensional
        hc = np.ravel ([casscf.fcisolver.contract_2e(h2cas, ci_i, ncas, nelecas, link_index=linkstrl) for ci_i in civec]).reshape (civec.shape[0], -1)
        return hc

    # part5
    jkcaa = numpy.empty((nocc,ncas))
    # part2, part3
    vhf_a = numpy.empty((nmo,nmo))
    # part1 ~ (J + 2K)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci0, ncas, nelecas, link_index=linkstr) # MRH: this should make the state-averaged density matrices
    dm2tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3)
    dm2tmp = dm2tmp.reshape(ncas**2,-1)
    hdm2 = numpy.empty((nmo,ncas,nmo,ncas))
    g_dm2 = numpy.empty((nmo,ncas))
    eri_cas = numpy.empty((ncas,ncas,ncas,ncas))
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        if i < nocc:
            jkcaa[i] = numpy.einsum('ik,ik->i', 6*kbuf[:,i]-2*jbuf[i], casdm1)
        vhf_a[i] =(numpy.einsum('quv,uv->q', jbuf, casdm1)
                 - numpy.einsum('uqv,uv->q', kbuf, casdm1) * .5)
        jtmp = lib.dot(jbuf.reshape(nmo,-1), casdm2.reshape(ncas*ncas,-1))
        jtmp = jtmp.reshape(nmo,ncas,ncas)
        ktmp = lib.dot(kbuf.transpose(1,0,2).reshape(nmo,-1), dm2tmp)
        hdm2[i] = (ktmp.reshape(nmo,ncas,ncas)+jtmp).transpose(1,0,2)
        g_dm2[i] = numpy.einsum('uuv->v', jtmp[ncore:nocc])
        if ncore <= i < nocc:
            eri_cas[i-ncore] = jbuf[ncore:nocc]
    jbuf = kbuf = jtmp = ktmp = dm2tmp = casdm2 = None
    vhf_ca = eris.vhf_c + vhf_a
    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

    ################# gradient #################
    gpq = numpy.zeros_like(h1e_mo)
    gpq[:,:ncore] = (h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
    gpq[:,ncore:nocc] = numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc],casdm1)
    gpq[:,ncore:nocc] += g_dm2

    h1cas_0 = h1e_mo[ncore:nocc,ncore:nocc] + eris.vhf_c[ncore:nocc,ncore:nocc]
    h2cas_0 = casscf.fcisolver.absorb_h1e(h1cas_0, eri_cas, ncas, nelecas, .5)
    hc0 = np.ravel ([casscf.fcisolver.contract_2e(h2cas_0,
        ci0_i, ncas, nelecas, link_index=linkstrl) for ci0_i in ci0]).reshape (nroot, -1)
    # MRH: equivalent to np.diagonal (np.dot (ci0.T, hc0)) but faster. Explicit broadcasting for the rare case where ndet = nroot
    eci0 = (hc0 * ci0).sum (1)[:,None]
    gci = hc0 - ci0 * eci0
    def g_update(u, fcivec):
        # MRH: fcivec should be for all roots
        fcivec = np.ravel (fcivec).reshape (nroot, -1)
        uc = u[:,:ncore].copy()
        ua = u[:,ncore:nocc].copy()
        rmat = u - numpy.eye(nmo)
        ra = rmat[:,ncore:nocc].copy()
        mo1 = numpy.dot(mo, u)
        mo_c = numpy.dot(mo, uc)
        mo_a = numpy.dot(mo, ua)
        dm_c = numpy.dot(mo_c, mo_c.T) * 2

        # MRH: I **think** this is how the axis kwarg works. Again I should explicitly broadcast it for the ndet = nroot edge case
        fcivec *= 1./numpy.linalg.norm(fcivec, axis=1)[:,None]
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas, link_index=linkstr) # MRH: state-averaged density matrices
        #casscf.with_dep4 = False
        #casscf.ci_response_space = 3
        #casscf.ci_grad_trust_region = 3
        #casdm1, casdm2, gci, fcivec = casscf.update_casdm(mo, u, fcivec, 0, eris, locals())
        dm_a = reduce(numpy.dot, (mo_a, casdm1, mo_a.T))
        vj, vk = casscf.get_jk(casscf.mol, (dm_c, dm_a))
        vhf_c = reduce(numpy.dot, (mo1.T, vj[0]-vk[0]*.5, mo1[:,:nocc]))
        vhf_a = reduce(numpy.dot, (mo1.T, vj[1]-vk[1]*.5, mo1[:,:nocc]))
        h1e_mo1 = reduce(numpy.dot, (u.T, h1e_mo, u[:,:nocc]))
        p1aa = numpy.empty((nmo,ncas,ncas*ncas))
        paa1 = numpy.empty((nmo,ncas*ncas,ncas))
        aaaa = numpy.empty([ncas]*4)
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            p1aa[i] = lib.dot(ua.T, jbuf.reshape(nmo,-1))
            paa1[i] = lib.dot(kbuf.transpose(0,2,1).reshape(-1,nmo), ra)
            if ncore <= i < nocc:
                aaaa[i-ncore] = jbuf[ncore:nocc]

# active space Hamiltonian up to 2nd order
        aa11 = lib.dot(ua.T, p1aa.reshape(nmo,-1)).reshape([ncas]*4)
        aa11 = aa11 + aa11.transpose(2,3,0,1) - aaaa
        a11a = lib.dot(ra.T, paa1.reshape(nmo,-1)).reshape((ncas,)*4)
        a11a = a11a + a11a.transpose(1,0,2,3)
        a11a = a11a + a11a.transpose(0,1,3,2)
        eri_cas_2 = aa11 + a11a
        h1cas_2 = h1e_mo1[ncore:nocc,ncore:nocc] + vhf_c[ncore:nocc,ncore:nocc]
        # MRH: contiguous in memory but two-dimensional
        fcivec = np.ravel (fcivec).reshape (nroot, -1) 
        hc0 = fci_matvec(fcivec, h1cas_2, eri_cas_2)
        # MRH: see eci0 line above. Changing name from "gci" to "my_gci" just to head off any possibility of name-shadowing causing a problem
        my_gci = hc0 - fcivec * (fcivec * hc0).sum (1)[:,None] 

        g = numpy.zeros_like(h1e_mo)
        g[:,:ncore] = (h1e_mo1[:,:ncore] + vhf_c[:,:ncore] + vhf_a[:,:ncore]) * 2
        g[:,ncore:nocc] = numpy.dot(h1e_mo1[:,ncore:nocc]+vhf_c[:,ncore:nocc], casdm1)
# 0000 + 1000 + 0100 + 0010 + 0001 + 1100 + 1010 + 1001  (missing 0110 + 0101 + 0011)
        p1aa = lib.dot(u.T, p1aa.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        paa1 = lib.dot(u.T, paa1.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0,1,3,2)
        g[:,ncore:nocc] += numpy.einsum('puwx,wxuv->pv', p1aa, casdm2)
        g_orb = casscf.pack_uniq_var(g-g.T)
        return numpy.hstack((g_orb*2, (my_gci * weights_2d).ravel ()*2))

    ############## hessian, diagonal ###########

    # part7
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    h_diag = numpy.einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = gpq.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    idx = numpy.arange(nmo)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal() # (pr|kl) * E(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1) * 2
    h_diag[:ncore] += v_diag * 2
    idx = numpy.arange(ncore)
    h_diag[idx,idx] -= v_diag[:ncore] * 4
    # V_{pr} E_{sq}
    tmp = numpy.einsum('ii,jj->ij', eris.vhf_c, casdm1)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -eris.vhf_c[ncore:nocc,ncore:nocc] * casdm1
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    # -2(pr|sq) + 4(pq|sr) + 4(pq|rs) - 2(ps|rq)
    tmp = 6 * eris.k_pc - 2 * eris.j_pc
    h_diag[ncore:,:ncore] += tmp[ncore:]
    h_diag[:ncore,ncore:] += tmp[ncore:].T

    # part5 and part6 diag
    # -(qr|kp) E_s^k  p in core, sk in active
    h_diag[:nocc,ncore:nocc] -= jkcaa
    h_diag[ncore:nocc,:nocc] -= jkcaa.T

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag.T
    h_diag[:,ncore:nocc] += v_diag

# Does this term contribute to internal rotation?
#    h_diag[ncore:nocc,ncore:nocc] -= v_diag[:,ncore:nocc]*2
    h_diag = casscf.pack_uniq_var(h_diag)

    # MRH: explicit broadcasting of hci_diag is the opposite of broadcasting of eci0
    # Therefore I should start with the last term, which has the correct dimensionality
    hci_diag = -gci * ci0 * 4
    hci_diag += casscf.fcisolver.make_hdiag(h1cas_0, eri_cas, ncas, nelecas)[None,:]
    hci_diag -= eci0
    hdiag_all = numpy.hstack((h_diag*2, hci_diag.ravel ()*2))

    g_orb = casscf.pack_uniq_var(gpq-gpq.T)
    g_all = numpy.hstack((g_orb*2, (gci * weights_2d).ravel ()*2))
    ngorb = g_orb.size

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x[:ngorb])
        # MRH: contiguous in memory but two-dimensional. Does this function see nroot? It must because it sees ngorb
        ci1 = x[ngorb:].reshape (nroot, -1)

        # H_cc
        # hc0, ci0, eci0, and ci1 should all have the right shape by now. hci1 needs to be explicitly broadcasted
        # Wait a second, isn't (hc0 - ci0*eci0) just gci?
        hci1 = np.ravel ([casscf.fcisolver.contract_2e(h2cas_0, ci1_i, ncas, nelecas, link_index=linkstrl) for ci1_i in ci1]).reshape (nroot, -1)
        hci1 -= (ci1 * eci0).sum (1)[:,None]
        #hci1 -= ((hc0-ci0*eci0)*ci0.dot(ci1) + ci0*(hc0-ci0*eci0).dot(ci1)) * 2
        hci1 -= (gci * (ci0 * ci1).sum (1)[:,None] + ci0 * (gci * ci1).sum (1)[:,None]) * 2

        # H_co
        # MRH: the trickiest part. I need transition density matrices for EACH state, NOT the average
        # MRH: as ugly as it is, I think an explicit for loop is the safest bet for this.
        # MRH: g_dm2 and vhf_a are used in other parts!  
        rc = x1[:,:ncore]
        ra = x1[:,ncore:nocc]
        ddm_c = numpy.zeros((nmo,nmo))
        ddm_c[:,:ncore] = rc[:,:ncore] * 2
        ddm_c[:ncore,:]+= rc[:,:ncore].T * 2
        vhf_a = numpy.empty((nroot, nmo,ncore))
        g_dm2 = numpy.empty((nroot, nmo,ncas))
        tdm1 = numpy.empty((nroot, ncas, ncas))
        tdm2 = numpy.empty((nroot, ncas, ncas, ncas, ncas))
        for iroot, (ci1_i, ci0_i) in enumerate (zip (ci1, ci0)):
            tdm1[iroot], tdm2[iroot] = casscf.fcisolver.trans_rdm12(ci1_i, ci0_i, ncas, nelecas, link_index=linkstr)
        tdm1 = tdm1 + tdm1.transpose(0,2,1)
        tdm2 = tdm2 + tdm2.transpose(0,2,1,4,3)
        tdm2 =(tdm2 + tdm2.transpose(0,3,4,1,2)) * .5
        paaa = numpy.empty((nmo,ncas,ncas,ncas))
        jk = 0
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            paaa[i] = jbuf[ncore:nocc]
            vhf_a[:,i,...] = numpy.einsum('quv,ruv->rq', jbuf[:ncore], tdm1)
            vhf_a[:,i,...]-= numpy.einsum('uqv,ruv->rq', kbuf[:,:ncore], tdm1) * .5
            jk += numpy.einsum('quv,q->uv', jbuf, ddm_c[i])
            jk -= numpy.einsum('uqv,q->uv', kbuf, ddm_c[i]) * .5
        g_dm2 = numpy.einsum('puwx,rwxuv->rpv', paaa, tdm2)
        aaaa = numpy.dot(ra.T, paaa.reshape(nmo,-1)).reshape([ncas]*4)
        aaaa = aaaa + aaaa.transpose(1,0,2,3)
        aaaa = aaaa + aaaa.transpose(2,3,0,1)
        h1aa = numpy.dot(h1e_mo[ncore:nocc]+eris.vhf_c[ncore:nocc], ra)
        h1aa = h1aa + h1aa.T + jk
        h1c0 = fci_matvec(ci0, h1aa, aaaa)
        hci1 += h1c0
        hci1 -= (ci0 * h1c0).sum (1)[:,None] * ci0 * weights_2d
        # MRH: I'm multiplying three-dimensional objects by weights from here on out
        tdm1 *= weights_3d
        tdm1 = tdm1.sum (0)
        g_dm2 *= weights_3d
        g_dm2 = g_dm2.sum (0)
        vhf_a *= weights_3d
        vhf_a = vhf_a.sum (0)

        # H_oo
        # part7
        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(lib.dot, (h1e_mo, x1, dm1))
        # part8
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        x2 -= numpy.dot((gpq+gpq.T), x1) * .5
        # part2
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += reduce(numpy.dot, (x1[:ncore,ncore:], vhf_ca[ncore:])) * 2
        # part3
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x1[ncore:nocc], eris.vhf_c))
        # part1
        x2[:,ncore:nocc] += numpy.einsum('purv,rv->pu', hdm2, x1[:,ncore:nocc])

        if ncore > 0:
            # part4, part5, part6
# Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
#    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
# x2[:,:ncore] += H * x1[:,:ncore] => (becuase x1=-x1.T) =>
# x2[:,:ncore] += -H' * x1[:ncore] => (becuase x2-x2.T) =>
# x2[:ncore] += H' * x1[:ncore]
            va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc

        # H_oc
        # MRH: Minimal broadcasting: I should have state-averaged all the relevant stuff up above except for s10
        s10 = (ci1 * ci0 * weights_2d).sum ()
        x2[:,:ncore] += ((h1e_mo[:,:ncore]+eris.vhf_c[:,:ncore]) * s10 + vhf_a) * 2
        x2[:,ncore:nocc] += numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc], tdm1)
        x2[:,ncore:nocc] += g_dm2
        x2 -= s10 * gpq 

        # (pr<->qs)
        x2 = x2 - x2.T
        return numpy.hstack((casscf.pack_uniq_var(x2)*2, hci1.ravel ()*2))

    return g_all, g_update, h_op, hdiag_all

def extract_rotation(casscf, dr, u, ci0):
    # MRH: I'm just gonna try to keep ci0 a 2d array while knowing that at any given moment it might not be
    nroot = len (ci0)
    ci0 = np.ravel (ci0).reshape (nroot, -1)
    ngorb = dr.size - ci0.size
    u = numpy.dot(u, casscf.update_rotate_matrix(dr[:ngorb]))
    ci1 = ci0 + dr[ngorb:].reshape (nroot, -1)
    ci1 *=1./numpy.linalg.norm(ci1, axis=1)[:,None]
    return u, ci1

def update_orb_ci(casscf, mo, ci0, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None:
        max_stepsize = casscf.max_stepsize

    nmo = mo.shape[1]
    nroot = len (ci0)
    ci0 = np.ravel (ci0).reshape (nroot, -1)
    g_all, g_update, h_op, h_diag = gen_g_hop(casscf, mo, ci0, eris)
    ngorb = g_all.size - ci0.size
    g_kf = g_all
    norm_gkf = norm_gall = numpy.linalg.norm(g_all)
    log.debug('    |g|=%5.3g (%4.3g %4.3g) (keyframe)', norm_gall,
              numpy.linalg.norm(g_all[:ngorb]),
              numpy.linalg.norm(g_all[ngorb:]))

    def precond(x, e):
        if callable(h_diag):
            x = h_diag(x, e-casscf.ah_level_shift)
        else:
            hdiagd = h_diag-(e-casscf.ah_level_shift)
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            x = x/hdiagd
        x *= 1/numpy.linalg.norm(x)
        return x

    def scale_down_step(dxi, hdxi):
        dxmax = abs(dxi).max()
        if dxmax > casscf.max_stepsize:
            scale = casscf.max_stepsize / dxmax
            log.debug1('Scale rotation by %g', scale)
            dxi *= scale
            hdxi *= scale
        return dxi, hdxi

    class Statistic:
        def __init__(self):
            self.imic = 0
            self.tot_hop = 0
            self.tot_kf = 1  # The call to gen_g_hop

    if x0_guess is None:
        x0_guess = g_all
    g_op = lambda: g_all

    stat = Statistic()
    dr = 0
    ikf = 0
    u = numpy.eye(nmo)
    ci_kf = ci0

    if norm_gall < conv_tol_grad*.3:
        return u, ci_kf, norm_gall, stat, x0_guess

    for ah_conv, ihop, w, dxi, hdxi, residual, seig \
            in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                                lindep=casscf.ah_lindep, verbose=log):
        stat.tot_hop = ihop
        norm_residual = numpy.linalg.norm(residual)
        if (ah_conv or ihop == casscf.ah_max_cycle or # make sure to use the last step
            ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or
            (seig < casscf.ah_lindep)):
            stat.imic += 1
            dxmax = abs(dxi).max()
            dxi, hdxi = scale_down_step(dxi, hdxi)

            dr += dxi
            g_all = g_all + hdxi
            norm_dr = numpy.linalg.norm(dr)
            norm_gall = numpy.linalg.norm(g_all)
            norm_gorb = numpy.linalg.norm(g_all[:ngorb])
            norm_gci = numpy.linalg.norm(g_all[ngorb:])
            log.debug('    imic %d(%d)  |g|=%3.2e (%2.1e %2.1e)  |dxi|=%3.2e '
                      'max(x)=%3.2e |dr|=%3.2e  eig=%2.1e seig=%2.1e',
                      stat.imic, ihop, norm_gall, norm_gorb, norm_gci, numpy.linalg.norm(dxi),
                      dxmax, norm_dr, w, seig)

            max_cycle = max(casscf.max_cycle_micro,
                            casscf.max_cycle_micro-int(numpy.log(norm_gkf+1e-7)*2))
            log.debug1('Set max_cycle %d', max_cycle)
            ikf += 1
            if stat.imic > 3 and norm_gall > norm_gkf*casscf.ah_grad_trust_region:
                g_all = g_all - hdxi
                dr -= dxi
                norm_gall = numpy.linalg.norm(g_all)
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (stat.imic >= max_cycle or norm_gall < conv_tol_grad*.3):
                break

            elif ((ikf >= max(casscf.kf_interval, casscf.kf_interval-numpy.log(norm_dr+1e-7)) or
# Insert keyframe if the keyframe and the esitimated grad are too different
                   norm_gall < norm_gkf/casscf.kf_trust_region)):
                ikf = 0
                u, ci_kf = extract_rotation(casscf, dr, u, ci_kf)
                dr[:] = 0
                g_kf1 = g_update(u, ci_kf)
                stat.tot_kf += 1
                norm_gkf1 = numpy.linalg.norm(g_kf1)
                norm_gorb = numpy.linalg.norm(g_kf1[:ngorb])
                norm_gci = numpy.linalg.norm(g_kf1[ngorb:])
                norm_dg = numpy.linalg.norm(g_kf1-g_all)
                log.debug('Adjust keyframe to |g|= %4.3g (%4.3g %4.3g) '
                          '|g-correction|= %4.3g',
                          norm_gkf1, norm_gorb, norm_gci, norm_dg)

                if (norm_dg < norm_gall*casscf.ah_grad_trust_region  # kf not too diff
                    #or norm_gkf1 < norm_gkf  # grad is decaying
                    # close to solution
                    or norm_gkf1 < conv_tol_grad*casscf.ah_grad_trust_region):
                    g_all = g_kf = g_kf1
                    g_kf1 = None
                    norm_gall = norm_gkf = norm_gkf1
                else:
                    g_all = g_all - hdxi
                    dr -= dxi
                    norm_gall = norm_gkf = numpy.linalg.norm(g_all)
                    log.debug('Out of trust region. Restore previouse step')
                    break

    u, ci_kf = extract_rotation(casscf, dr, u, ci_kf)
    log.debug('    tot inner=%d  |g|= %4.3g (%4.3g %4.3g) |u-1|= %4.3g  |dci|= %4.3g',
              stat.imic, norm_gall, norm_gorb, norm_gci,
              numpy.linalg.norm(u-numpy.eye(nmo)),
              numpy.linalg.norm(ci_kf-ci0))
    return u, ci_kf, norm_gkf, stat, dxi


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''Second order CASSCF driver
    '''
    log = logger.new_logger(casscf, verbose)
    log.warn('SO-CASSCF (Second order CASSCF) is an experimental feature. '
             'Its performance is bad for large systems.')

    cput0 = (time.clock(), time.time())
    log.debug('Start SO-CASSCF (newton CASSCF)')
    if callback is None:
        callback = casscf.callback

    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, casscf.ncas, casscf.nelecas)
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    dr0 = None

    t2m = t1m = log.timer('Initializing newton CASSCF', *cput0)
    imacro = 0
    tot_hop = 0
    tot_kf  = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        u, fcivec, norm_gall, stat, dr0 = \
                update_orb_ci(casscf, mo, fcivec, eris, dr0, conv_tol_grad*.3, verbose=log)
        tot_hop += stat.tot_hop
        tot_kf  += stat.tot_kf
        t2m = log.timer('update_orb_ci', *t2m)

        eris = None
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t2m)

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and norm_gall < conv_tol_grad):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('newton CASSCF converged in %d macro (%d KF %d Hx) steps',
                 imacro, tot_kf, tot_hop)
    else:
        log.info('newton CASSCF not converged, %d macro (%d KF %d Hx) steps',
                 imacro, tot_kf, tot_hop)

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, casscf.ncas, casscf.nelecas)
    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb: # dump_chk may save casdm1
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = -occ

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('newton CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy







