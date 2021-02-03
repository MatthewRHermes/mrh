import numpy as np
import scipy, time
from pyscf import lib, mcscf
from pyscf.mcscf.mc1step import _fake_h_for_fast_casci
from mrh.my_pyscf.mcpdft import mcpdft

def get_heff_cas (mc, mo_coeff, ci):
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    
    veff1, veff2 = mc.get_pdft_veff (mo=mo_coeff, ci=ci, incl_coul=False, paaa_only=True)
    dm_core = 2 * mo_core @ mo_core.conj ().T
    vj_core = mc._scf.get_j (dm=dm_core)
    h1 = mo_cas.conj ().T @ (mc.get_hcore () + veff1 + vj_core) @ mo_cas
    h1 += veff2.vhf_c[ncore:nocc,ncore:nocc]
    h2 = np.zeros ([ncas,]*4)
    for i in range (ncore, nocc): 
        h2[i-ncore] = veff2.ppaa[i][ncore:nocc].copy ()
    return 0.0, h1, h2

def casci(mc, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    ''' Minimize the PDFT energy expression for a single state by
    varying the vector using a fixed-point FCI algorithm '''
    if ci0 is None: ci0 = mc.ci
    t0 = (time.clock (), time.time ())
    ncas, nelecas = mc.ncas, mc.nelecas
    linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    linkstr  = mc.fcisolver.gen_linkstr(ncas, nelecas, False)
    log = lib.logger.new_logger (mc, mc.verbose)
    max_memory = max(400, mc.max_memory-lib.current_memory()[0])
    if eris is None:
        h0_cas, h1_cas = mcscf.casci.h1e_for_cas (mc, mo_coeff, mc.ncas, mc.ncore)
        h2_cas = mcscf.casci.CASCI.ao2mo (mc, mo_coeff)
    else:
        fcasci = _fake_h_for_fast_casci (mc, mo_coeff, eris)
        h1_cas, h0_cas = fcasci.get_h1eff ()
        h2_cas = fcasci.get_h2eff ()

    if ci0 is None: 
        # Use real Hamiltonian? Or use HF?
        hdiag = mc.fcisolver.make_hdiag (h1_cas, h2_cas, mc.ncas, mc.nelecas)
        ci0 = mc.fcisolver.get_init_guess (ncas, nelecas, 1, hdiag)[0]

    ci1 = ci0.copy ()
    for it in range (mc.max_cycle_fp):
        with lib.temporary_env (mc, ci=ci1): # TODO: remove; fix calls
            h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo_coeff, ci1)
            h2eff = mc.fcisolver.absorb_h1e (h1_cas, h2_cas, ncas, nelecas, 0.5)
            hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
            ecas = ci1.conj ().ravel ().dot (hc) + h0_cas
            h2eff = mc.fcisolver.absorb_h1e (h1_pdft, h2_pdft, ncas, nelecas, 0.5)
            kc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
            ckc = ci1.conj ().ravel ().dot (kc)
            ci_grad = hc - (ckc * ci1.ravel ())
            ci_grad_norm = ci_grad.dot (ci_grad)
            epdft = mcpdft.kernel (mc, mc.otfnal, verbose=0)[0]
            lib.logger.info (mc, 'MC-PDFT CI fp iter %d ECAS = %e, EPDFT = %e, |grad| = %e, <c.K.c> = %e', it, ecas, epdft, ci_grad_norm, ckc)
            if ci_grad_norm < mc.conv_tol_ci_fp: break
       
            e_cas, ci1 = mc.fcisolver.kernel (h1_pdft, h2_pdft, ncas, nelecas,
                                              ci0=ci1, verbose=log,
                                              max_memory=max_memory,
                                              ecore=0)

    lib.logger.timer (mc, 'MC-PDFT CI fp iteration', *t0)
    if envs is not None and log.verbose >= lib.logger.INFO:
        log.debug('CAS space CI energy = %.15g', e_cas)

        if getattr(mc.fcisolver, 'spin_square', None):
            ss = mc.fcisolver.spin_square(ci1, mc.ncas, mc.nelecas)
        else:
            ss = None

        if 'imicro' in envs:  # Within CASSCF iteration
            if ss is None:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  CASCI E = %.15g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], ecas)
            else:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  S^2 = %.7f  CASCI E = %.15g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], ss[0], ecas)
            if 'norm_gci' in envs:
                log.info('               |grad[o]|=%5.3g  '
                         '|grad[c]|= %s  |ddm|=%5.3g',
                         envs['norm_gorb0'],
                         envs['norm_gci'], envs['norm_ddm'])
            else:
                log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                         envs['norm_gorb0'], envs['norm_ddm'])
        else:  # Initialization step
            if ss is None:
                log.info('MC-PDFT E = %.15g  CASCI E = %.15g', epdft, ecas)
            else:
                log.info('MC-PDFT E = %.15g  S^2 = %.7f  CASCI E = %.15g', epdft, ss[0], ecas)

    return epdft, ecas, ci1

def update_casdm(mc, mo, u, fcivec, e_cas, eris, envs={}):


