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
from pyscf.mcscf import casci, mc1step, newton_casscf
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import chkfile
from pyscf import ao2mo
from pyscf import scf
from pyscf.soscf import ciah
from pyscf import fci

# MRH: I keep writing "np" so I'll just set it here so I don't have to worry about it in either direction
np = numpy
from mrh.my_pyscf.mcscf import sacasscf
from mrh.my_pyscf.fci import csf, csf_symm
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det
from pyscf.fci.direct_spin1 import _unpack_nelec

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, ci0, eris, verbose=None):
# MRH: I'm just going to pass this to the newton_casscf version and average/weight/change to and from CSF's post facto!
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nelecas = casscf.nelecas
    neleca, nelecb = _unpack_nelec (nelecas)
    nmo = mo.shape[1]
    nroot = len (ci0)
    ndet = ci0[0].size
    norb = mo.shape[-1]
    ngorb = np.count_nonzero (casscf.uniq_var_indices (norb, ncore, ncas, casscf.frozen))
    weights_2d = np.asarray (casscf.weights)[:,None] 
    weights_3d = np.asarray (casscf.weights)[:,None,None] 
    mc_fci = casscf.fcisolver
    is_csf = isinstance (mc_fci, (csf.FCISolver, csf_symm.FCISolver,))
    if is_csf:
        if hasattr (mc_fci, 'wfnsym') and hasattr (mc_fci, 'confsym'):
            idx_sym = mc_fci.confsym[mc_fci.econf_csf_mask] == mc_fci.wfnsym
        else:
            idx_sym = None
        smult = mc_fci.smult

    fcasscf = mc1step.CASSCF (casscf._scf, ncas, nelecas)
    fcasscf.fcisolver = casscf.ss_fcisolver
    gh_roots = [newton_casscf.gen_g_hop (fcasscf, mo, ci0_i, eris, verbose=verbose) for ci0_i in ci0]

    def avg_orb_wgt_ci (x_roots):
        x_orb = sum ([x_iroot[:ngorb] * w for x_iroot, w in zip (x_roots, casscf.weights)])
        x_ci = np.stack ([x_iroot[ngorb:] * w for x_iroot, w in zip (x_roots, casscf.weights)], axis=0)
        if is_csf:
            x_ci = csf.pack_sym_ci (transform_civec_det2csf (x_ci, ncas, neleca, nelecb, smult,
                csd_mask=mc_fci.csd_mask, do_normalize=False)[0], idx_sym)
        x_all = np.append (x_orb, x_ci.ravel ()).ravel ()
        return x_all

    g_all = avg_orb_wgt_ci ([gh_iroot[0] for gh_iroot in gh_roots])
    hdiag_all = avg_orb_wgt_ci ([gh_iroot[3] for gh_iroot in gh_roots])

    def g_update (u, fcivec):
        return avg_orb_wgt_ci ([gh_iroot[1] (u, ci) for gh_iroot, ci in zip (gh_roots, fcivec)])

    def h_op (x):
        x_orb = x[:ngorb]
        x_ci = x[ngorb:].reshape (nroot, -1)
        if is_csf:
            x_ci = transform_civec_csf2det (csf.unpack_sym_ci (x_ci, idx_sym), ncas,
                neleca, nelecb, smult, csd_mask=mc_fci.csd_mask, do_normalize=False)[0]
        return avg_orb_wgt_ci ([gh_iroot[2] (np.append (x_orb, x_ci_iroot))
            for gh_iroot, x_ci_iroot in zip (gh_roots, x_ci)])

    return g_all, g_update, h_op, hdiag_all

def extract_rotation(casscf, dr, u, ci0):
    # MRH: I'm just gonna try to keep ci0 a 2d array while knowing that at any given moment it might not be
    nroot = len (ci0)
    ci0 = np.ravel (ci0).reshape (nroot, -1)
    ncas, ncore = casscf.ncas, casscf.ncore
    norb = u.shape[-1]
    nocc = ncore + ncas
    ngorb = np.count_nonzero (casscf.uniq_var_indices (norb, ncore, ncas, casscf.frozen))
    u = numpy.dot(u, casscf.update_rotate_matrix(dr[:ngorb]))
    dci = dr[ngorb:].reshape (nroot, -1)
    if isinstance (casscf.fcisolver, (csf.FCISolver, csf_symm.FCISolver,)):
        mc_fci = casscf.fcisolver
        if hasattr (mc_fci, 'wfnsym') and hasattr (mc_fci, 'confsym'):
            idx_sym = mc_fci.confsym[mc_fci.econf_csf_mask] == mc_fci.wfnsym
        else:
            idx_sym = None
        smult = mc_fci.smult
        neleca, nelecb = _unpack_nelec (casscf.nelecas)
        dci = transform_civec_csf2det (csf.unpack_sym_ci (dci, idx_sym), ncas, neleca, nelecb, smult,
            csd_mask=mc_fci.csd_mask, do_normalize=False)[0]
    ci1 = ci0 + dci
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
            hdiagd = h_diag-(e+casscf.ah_level_shift)
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







