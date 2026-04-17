import inspect
from functools import reduce
from itertools import product
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import casci, mc1step, addons
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf import scf
from pyscf.soscf import ciah

def davidson_cc (mc, h_op, g_all, x0_guess, precond, tol=None, g_update=None, callback=None,
                 verbose=None, conv_tol_grad=None, **kwargs):
    log = logger.new_logger(mc, verbose)
    if tol is None:
        tol = mc.ah_conv_tol
    norm_gkf = norm_gall = np.linalg.norm(g_all)
    log.debug('    |g|=%5.3g (keyframe)', norm_gall)

    if len (inspect.getfullargspec(precond).args)==2:
        prec_op = precond
        def precond (x, e):
            return prec_op (x)

    def scale_down_step(dxi, hdxi):
        dxmax = abs(dxi).max()
        if dxmax > mc.max_stepsize:
            scale = mc.max_stepsize / dxmax
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
    dr = np.zeros_like (g_all)
    ikf = 0
    dr_list = []

    if norm_gall < conv_tol_grad*.3:
        return dr, g_all, stat

    for ah_conv, ihop, w, dxi, hdxi, residual, seig \
            in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=tol, max_cycle=mc.ah_max_cycle,
                                lindep=mc.ah_lindep, verbose=log):
        stat.tot_hop = ihop
        norm_residual = np.linalg.norm(residual)
        if (ah_conv or ihop == mc.ah_max_cycle or # make sure to use the last step
            ((norm_residual < mc.ah_start_tol) and (ihop >= mc.ah_start_cycle)) or
            (seig < mc.ah_lindep)):
            stat.imic += 1
            dxmax = abs(dxi).max()
            dxi, hdxi = scale_down_step(dxi, hdxi)

            dr += dxi
            g_all = g_all + hdxi
            norm_dr = np.linalg.norm(dr)
            norm_gall = np.linalg.norm(g_all)
            log.debug('    imic %d(%d)  |g|=%3.2e |dxi|=%3.2e '
                      'max(x)=%3.2e |dr|=%3.2e  eig=%2.1e seig=%2.1e',
                      stat.imic, ihop, norm_gall, np.linalg.norm(dxi),
                      dxmax, norm_dr, w, seig)

            max_cycle = max(mc.max_cycle_micro,
                            mc.max_cycle_micro-int(np.log(norm_gkf+1e-7)*2))
            log.debug1('Set max_cycle %d', max_cycle)
            ikf += 1
            if stat.imic > 3 and norm_gall > norm_gkf*mc.ah_grad_trust_region:
                g_all = g_all - hdxi
                dr -= dxi
                norm_gall = np.linalg.norm(g_all)
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (stat.imic >= max_cycle or norm_gall < conv_tol_grad*.3):
                break

            # TODO: implement g_update and restore original logic
            elif (ikf >= max(mc.kf_interval, mc.kf_interval-np.log(norm_dr+1e-7)) or
                  # Insert keyframe if the keyframe and the estimated grad are too different
                  (norm_gall < norm_gkf/mc.kf_trust_region)):
                if not callable (g_update):
                    log.debug('Out of trust region. Restore previouse step')
                    g_all = g_all - hdxi
                    dr -= dxi
                    norm_gall = norm_gkf = np.linalg.norm(g_all)
                    break
                ikf = 0
                dr_list.append (dr.copy ())
                g_kf1 = g_update (dr_list)
                dr[:] = 0
                stat.tot_kf += 1
                norm_gkf1 = np.linalg.norm(g_kf1)
                norm_dg = np.linalg.norm(g_kf1-g_all)
                log.debug('Adjust keyframe to |g|= %4.3g '
                          '|g-correction|= %4.3g',
                          norm_gkf1, norm_dg)

                log.debug ('%f %f %f %f %f', norm_dg, norm_gall, norm_gkf1, conv_tol_grad, mc.ah_grad_trust_region)
                if (norm_dg < norm_gall*mc.ah_grad_trust_region  # kf not too diff
                    #or norm_gkf1 < norm_gkf  # grad is decaying
                    # close to solution
                    or norm_gkf1 < conv_tol_grad*mc.ah_grad_trust_region):
                    log.debug('kf not too diff')
                    g_all = g_kf1
                    g_kf1 = None
                    norm_gall = norm_gkf = norm_gkf1
                # TODO: understand why this new branch is necessary for anything to converge!
                # Does it have something to do with the preconditioner??
                elif norm_gkf1 > norm_gall * mc.ah_grad_trust_region:
                    log.debug ('Hessian breakdown; iterate')
                    break
                else:
                    if stat.imic > 1:
                        g_all = g_all - hdxi
                        dr -= dxi
                        log.debug('Out of trust region. Restore previouse step')
                    else:
                        log.debug('Out of trust region but I need at least one step')
                    norm_gall = norm_gkf = np.linalg.norm(g_all)
                    break

            if callable (callback):
                callback (dr)

    if len (dr_list) > 0:
        dr_list.append (dr)
        dr = dr_list
    log.debug('    tot inner=%d  |g|= %4.3g',
              stat.imic, norm_gall) 
    return dr, g_all, stat

